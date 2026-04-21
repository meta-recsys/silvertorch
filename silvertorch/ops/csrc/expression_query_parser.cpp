// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "expression_query_parser.h"
#include <torch/library.h>
#include <torch/torch.h>
#include "bloom_index_util.cuh"
#include "bloom_index_util.h"

namespace st::ops::expression_query_parser {

using at::Tensor;
using namespace st::ops::bloom_index;

namespace {

// --- Tokenizer ---

enum class TokenType {
  NUMBER,
  FLOAT_NUM,
  COLON,
  LPAREN,
  RPAREN,
  AND,
  OR,
  NOT,
  END,
};

struct Token {
  TokenType type;
  std::string value;
  size_t pos;
};

class Tokenizer {
 public:
  explicit Tokenizer(const std::string& input) : input_(input), pos_(0) {}

  Token next() {
    skipWhitespace();
    if (pos_ >= input_.size()) {
      return {TokenType::END, "", pos_};
    }
    size_t start = pos_;
    char c = input_[pos_];

    switch (c) {
      case '(':
        pos_++;
        return {TokenType::LPAREN, "(", start};
      case ')':
        pos_++;
        return {TokenType::RPAREN, ")", start};
      case ':':
        pos_++;
        return {TokenType::COLON, ":", start};
      case '&':
        pos_++;
        return {TokenType::AND, "&", start};
      case '|':
        pos_++;
        return {TokenType::OR, "|", start};
      case '!':
        pos_++;
        return {TokenType::NOT, "!", start};
      default:
        break;
    }

    if (isNumberStart(c)) {
      return readNumber(start);
    }

    if (std::isalpha(c)) {
      return readKeyword(start);
    }

    TORCH_CHECK(false, "Unexpected character '", c, "' at position ", start);
  }

  Token peek() {
    size_t saved = pos_;
    Token t = next();
    pos_ = saved;
    return t;
  }

 private:
  bool isNumberStart(char c) const {
    return std::isdigit(c) ||
        (c == '-' && pos_ + 1 < input_.size() &&
         std::isdigit(input_[pos_ + 1]));
  }

  void skipWhitespace() {
    while (pos_ < input_.size() && std::isspace(input_[pos_])) {
      pos_++;
    }
  }

  Token readNumber(size_t start) {
    std::string num;
    if (input_[pos_] == '-') {
      num += input_[pos_++];
    }
    bool has_dot = false;
    while (pos_ < input_.size() &&
           (std::isdigit(input_[pos_]) || input_[pos_] == '.')) {
      if (input_[pos_] == '.') {
        if (has_dot) {
          break;
        }
        has_dot = true;
      }
      num += input_[pos_++];
    }
    return {
        has_dot ? TokenType::FLOAT_NUM : TokenType::NUMBER,
        std::move(num),
        start};
  }

  Token readKeyword(size_t start) {
    std::string word;
    while (pos_ < input_.size() && std::isalpha(input_[pos_])) {
      word += input_[pos_++];
    }
    if (word == "AND") {
      return {TokenType::AND, std::move(word), start};
    }
    if (word == "OR") {
      return {TokenType::OR, std::move(word), start};
    }
    if (word == "NOT") {
      return {TokenType::NOT, std::move(word), start};
    }
    TORCH_CHECK(false, "Unknown keyword '", word, "' at position ", start);
  }

  std::string input_;
  size_t pos_;
};

// --- Plan Builder ---
// Recursive descent parser that directly builds QueryPlan.
// Replicates the same traversal and plan construction as
// bloom_index_util.cpp::parse(), but driven by text tokens.
//
// Grammar (precedence: NOT > AND > OR):
//   expression  := or_expr
//   or_expr     := and_expr (('OR' | '|') and_expr)*
//   and_expr    := unary_expr (('AND' | '&') unary_expr)*
//   unary_expr  := ('NOT' | '!') unary_expr | primary
//   primary     := '(' expression ')' | feature_term
//   feature_term := NUMBER ':' NUMBER (':' NUMBER)?

class ExpressionPlanBuilder {
 public:
  ExpressionPlanBuilder(
      const std::string& input,
      QueryPlan<true>& plan,
      int64_t k,
      int64_t max_sub_queries)
      : tokenizer_(input),
        plan_(plan),
        k_(k),
        max_sub_queries_(max_sub_queries) {}

  void build() {
    buildOrExpr();
    auto tok = tokenizer_.peek();
    TORCH_CHECK(
        tok.type == TokenType::END,
        "Unexpected token '",
        tok.value,
        "' at position ",
        tok.pos,
        ", expected end of expression");
  }

 private:
  void emitCompoundOp(Operator op, std::vector<int32_t>& sub_operator_pos) {
    plan_.parameters.insert(
        plan_.parameters.end(),
        sub_operator_pos.begin(),
        sub_operator_pos.end());
    plan_.operators.push_back(op);
    // NOLINTNEXTLINE(facebook-hte-MemberUncheckedArrayBounds)
    plan_.offsets.push_back(
        plan_.offsets.back() + static_cast<int32_t>(sub_operator_pos.size()));
  }

  void buildOrExpr() {
    std::vector<int32_t> sub_operator_pos;
    int64_t existing_queries = 0;
    bool batching_occurred = false;

    auto maybeBatch = [&]() {
      if (existing_queries == max_sub_queries_) {
        emitCompoundOp(Operator::OR, sub_operator_pos);
        sub_operator_pos.clear();
        sub_operator_pos.push_back(
            static_cast<int32_t>(plan_.operators.size()) - 1);
        existing_queries = 1;
        batching_occurred = true;
      }
    };

    buildAndExpr();
    sub_operator_pos.push_back(
        static_cast<int32_t>(plan_.operators.size()) - 1);
    existing_queries++;
    // Only batch the first operand if there are more OR operands coming
    if (tokenizer_.peek().type == TokenType::OR) {
      maybeBatch();
    }

    while (tokenizer_.peek().type == TokenType::OR) {
      tokenizer_.next();
      buildAndExpr();
      sub_operator_pos.push_back(
          static_cast<int32_t>(plan_.operators.size()) - 1);
      existing_queries++;
      maybeBatch();
    }

    if (sub_operator_pos.size() <= 1 && !batching_occurred) {
      return;
    }
    emitCompoundOp(Operator::OR, sub_operator_pos);
  }

  void buildAndExpr() {
    std::vector<int32_t> sub_operator_pos;
    int64_t existing_queries = 0;
    bool batching_occurred = false;

    auto maybeBatch = [&]() {
      if (existing_queries == max_sub_queries_) {
        emitCompoundOp(Operator::AND, sub_operator_pos);
        sub_operator_pos.clear();
        sub_operator_pos.push_back(
            static_cast<int32_t>(plan_.operators.size()) - 1);
        existing_queries = 1;
        batching_occurred = true;
      }
    };

    buildUnaryExpr();
    sub_operator_pos.push_back(
        static_cast<int32_t>(plan_.operators.size()) - 1);
    existing_queries++;
    // Only batch the first operand if there are more AND operands coming
    if (tokenizer_.peek().type == TokenType::AND) {
      maybeBatch();
    }

    while (tokenizer_.peek().type == TokenType::AND) {
      tokenizer_.next();
      buildUnaryExpr();
      sub_operator_pos.push_back(
          static_cast<int32_t>(plan_.operators.size()) - 1);
      existing_queries++;
      maybeBatch();
    }

    if (sub_operator_pos.size() <= 1 && !batching_occurred) {
      return;
    }
    emitCompoundOp(Operator::AND, sub_operator_pos);
  }

  void buildUnaryExpr() {
    if (tokenizer_.peek().type == TokenType::NOT) {
      tokenizer_.next();
      buildUnaryExpr();

      std::vector<int32_t> sub_operator_pos;
      sub_operator_pos.push_back(
          static_cast<int32_t>(plan_.operators.size()) - 1);
      emitCompoundOp(Operator::NOT, sub_operator_pos);
      return;
    }
    buildPrimary();
  }

  void buildPrimary() {
    if (tokenizer_.peek().type == TokenType::LPAREN) {
      tokenizer_.next();
      buildOrExpr();
      auto tok = tokenizer_.next();
      TORCH_CHECK(
          tok.type == TokenType::RPAREN,
          "Expected ')' at position ",
          tok.pos,
          ", got '",
          tok.value,
          "'");
      return;
    }
    buildFeatureTerm();
  }

  void buildFeatureTerm() {
    auto id_tok = tokenizer_.next();
    TORCH_CHECK(
        id_tok.type == TokenType::NUMBER,
        "Expected feature id (integer) at position ",
        id_tok.pos,
        ", got '",
        id_tok.value,
        "'");

    auto colon1 = tokenizer_.next();
    TORCH_CHECK(
        colon1.type == TokenType::COLON,
        "Expected ':' after feature id at position ",
        colon1.pos);

    auto value_tok = tokenizer_.next();
    TORCH_CHECK(
        value_tok.type == TokenType::NUMBER,
        "Expected feature value (integer) at position ",
        value_tok.pos,
        ", got '",
        value_tok.value,
        "'");

    int64_t feature_id = std::stoll(id_tok.value);
    int64_t feature_value = std::stoll(value_tok.value);

    // Skip optional weight (id:value:weight) — weight is not used in plan
    if (tokenizer_.peek().type == TokenType::COLON) {
      tokenizer_.next();
      auto weight_tok = tokenizer_.next();
      TORCH_CHECK(
          weight_tok.type == TokenType::FLOAT_NUM ||
              weight_tok.type == TokenType::NUMBER,
          "Expected weight (number) at position ",
          weight_tok.pos,
          ", got '",
          weight_tok.value,
          "'");
    }

    // Compute oneBitsPosition hashes (bloom v2 path)
    std::vector<QueryPlanOneBitsPType<true>::type> one_bits_position(k_, 0);
    prefetch_one_bits_hashes(
        feature_id, feature_value, k_, one_bits_position.data());

    plan_.oneBitsPosition.insert(
        plan_.oneBitsPosition.end(),
        one_bits_position.begin(),
        one_bits_position.end());

    plan_.operators.push_back(Operator::TERM);
    // NOLINTNEXTLINE(facebook-hte-MemberUncheckedArrayBounds)
    plan_.offsets.push_back(plan_.offsets.back() + 1);
    plan_.parameters.push_back(
        static_cast<int32_t>(plan_.oneBitsPosition.size() / k_) - 1);
  }

  Tokenizer tokenizer_;
  QueryPlan<true>& plan_;
  int64_t k_;
  int64_t max_sub_queries_;
};

template <typename native_type, at::ScalarType aten_type>
inline at::Tensor vec_to_tensor(std::vector<native_type>& vec) {
  return torch::from_blob(
             vec.data(), {static_cast<int32_t>(vec.size())}, aten_type)
      .clone();
}

} // namespace

// --- Public API ---

std::tuple<int64_t, std::vector<Tensor>> parse_expression_query_batch_cpu(
    const std::vector<std::string>& expressions,
    const Tensor& /*silvertorch_ks*/,
    int64_t bloom_hash_k,
    bool return_query_plan,
    int64_t max_sub_queries) {
  std::vector<QueryPlan<true>> query_plans;

  for (const auto& expr : expressions) {
    QueryPlan<true> plan;
    if (expr.empty()) {
      // Empty expression → EMPTY operator (match all)
      plan.operators.push_back(Operator::EMPTY);
      plan.offsets.push_back(plan.offsets.back() + 0);
    } else {
      ExpressionPlanBuilder builder(expr, plan, bloom_hash_k, max_sub_queries);
      builder.build();
    }
    query_plans.push_back(std::move(plan));
  }

  std::vector<char> query_plan_data_vec;
  std::vector<size_t> query_plan_offsets_vec;
  uint32_t max_stack_size = 0;

  if (return_query_plan) {
    max_stack_size = encode_query_plans<true>(
        query_plans, query_plan_data_vec, query_plan_offsets_vec);
  } else {
    max_stack_size = get_max_stack_size<true>(query_plans);
  }

  std::vector<Tensor> result = {
      vec_to_tensor<char, at::kChar>(query_plan_data_vec),
      vec_to_tensor<size_t, torch::kInt64>(query_plan_offsets_vec)};
  return std::make_tuple(
      static_cast<int64_t>(max_stack_size), std::move(result));
}

namespace {

// Torch op wrapper: converts c10::List<std::string> to std::vector<std::string>
std::tuple<int64_t, std::vector<Tensor>> parse_expression_query_batch_op(
    const c10::List<std::string>& expressions,
    const Tensor& silvertorch_ks,
    int64_t bloom_hash_k,
    bool return_query_plan,
    int64_t max_sub_queries) {
  std::vector<std::string> expr_strs;
  expr_strs.reserve(expressions.size());
  for (size_t i = 0; i < expressions.size(); ++i) {
    expr_strs.push_back(expressions.get(i));
  }
  return parse_expression_query_batch_cpu(
      expr_strs,
      silvertorch_ks,
      bloom_hash_k,
      return_query_plan,
      max_sub_queries);
}

} // namespace

TORCH_LIBRARY_FRAGMENT(st, m) {
  m.def(
      "parse_expression_query_batch("
      "str[] expressions, "
      "Tensor silvertorch_ks, "
      "int bloom_hash_k, "
      "bool return_query_plan=True, "
      "int max_sub_queries=5)"
      "-> (int, Tensor[])");
}

TORCH_LIBRARY_IMPL(st, CPU, m) {
  m.impl(
      "parse_expression_query_batch",
      torch::dispatch(
          c10::DispatchKey::CPU, TORCH_FN(parse_expression_query_batch_op)));
}

} // namespace st::ops::expression_query_parser

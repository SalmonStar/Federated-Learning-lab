// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_LEDGER_SESSION_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_LEDGER_SESSION_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "containers/fed_fl/inference_model.h"
#include "containers/fed_fl/session_utils.h"
#include "containers/session.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"

namespace confidential_federated_compute::fed_fl {

// FedSql implementation of Session interface. Not threadsafe.
class FedFlSession final
    : public confidential_federated_compute::LegacySession {

 public:
  FedFlSession(
      std::unique_ptr<tensorflow_federated::aggregation::CheckpointAggregator>
          aggregator,
      const std::vector<tensorflow_federated::aggregation::Intrinsic>& intrinsics,
      std::optional<SessionInferenceConfiguration> inference_configuration,
      const std::optional<uint32_t> serialize_output_access_policy_node_id,
      const std::optional<uint32_t> report_output_access_policy_node_id,
      absl::string_view sensitive_values_key);

  // FL container 不支援 SQL configure，直接 OK
  absl::Status ConfigureSession(
      fcp::confidentialcompute::SessionRequest configure_request) override {
    return absl::OkStatus();
  }

 private:
  std::unique_ptr<tensorflow_federated::aggregation::CheckpointAggregator>
      aggregator_;
  const std::vector<tensorflow_federated::aggregation::Intrinsic>& intrinsics_;

  InferenceModel inference_model_;
  const std::optional<uint32_t> serialize_output_access_policy_node_id_;
  const std::optional<uint32_t> report_output_access_policy_node_id_;

  absl::string_view sensitive_values_key_;
};

}  // namespace confidential_federated_compute::fed_fl

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_LEDGER_SESSION_H_

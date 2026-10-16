# Licensed to the Software Freedom Conservancy (SFC) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The SFC licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# This file is generated from the WebDriver BiDi specification.
# DO NOT EDIT. Regenerate with:
#   bazel run //py:generate-bidi-protocol


from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from selenium.webdriver.common._bidi.domain import Domain
from selenium.webdriver.common._bidi.serialization import UNSET, Record, UnsetType, meta, register


@register("digitalCredentials.VirtualWalletAction")
class VirtualWalletAction(str, Enum):
    """digitalCredentials.VirtualWalletAction.

    See https://w3c-fedid.github.io/digital-credentials/#cddl-type-digitalcredentials-virtualwalletaction
    """

    DECLINE = "decline"
    RESPOND = "respond"
    WAIT = "wait"
    CLEAR = "clear"


@register("digitalCredentials.SetVirtualWalletBehaviorParameters")
@dataclass(frozen=True)
class SetVirtualWalletBehaviorParameters(Record):
    """digitalCredentials.SetVirtualWalletBehaviorParameters.

    See https://w3c-fedid.github.io/digital-credentials/#cddl-type-digitalcredentials-setvirtualwalletbehaviorparameters
    """

    action: VirtualWalletAction = field(
        metadata=meta("action", required=True, enum="digitalCredentials.VirtualWalletAction"),
    )
    context: str | UnsetType = field(default=UNSET, metadata=meta("context", primitive="str"))
    protocol: str | UnsetType = field(default=UNSET, metadata=meta("protocol", primitive="str"))
    response: Any | UnsetType = field(default=UNSET, metadata=meta("response"))


class DigitalCredentials(Domain):
    """Internal, unsupported.

    See https://www.selenium.dev/documentation/warnings/bidi-implementation/
    """

    def set_virtual_wallet_behavior(
        self,
        action: VirtualWalletAction,
        context: str | UnsetType = UNSET,
        protocol: str | UnsetType = UNSET,
        response: Any | UnsetType = UNSET,
    ) -> Any:
        """Execute digitalCredentials.setVirtualWalletBehavior (internal, unsupported)."""
        params = SetVirtualWalletBehaviorParameters(
            action=action,
            context=context,
            protocol=protocol,
            response=response,
        )
        return self._execute("digitalCredentials.setVirtualWalletBehavior", params=params, result=None)

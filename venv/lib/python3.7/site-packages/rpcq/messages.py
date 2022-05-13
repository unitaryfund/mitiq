#!/usr/bin/env python

"""
WARNING: This file is auto-generated, do not edit by hand. See README.md.
"""

import sys

from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional

if sys.version_info < (3, 7):
    from rpcq.external.dataclasses import dataclass, field, InitVar
else:
    from dataclasses import dataclass, field, InitVar


@dataclass(eq=False, repr=False)
class ParameterSpec(Message):
    """
    Specification of a dynamic parameter type and array-length.
    """

    type: str = ""
    """The parameter type, e.g., one of 'INTEGER', or 'FLOAT'."""

    length: int = 1
    """If this is not 1, the parameter is an array of this length."""


@dataclass(eq=False, repr=False)
class ParameterAref(Message):
    """
    A parametric expression.
    """

    name: str
    """The parameter name"""

    index: int
    """The array index."""


@dataclass(eq=False, repr=False)
class PatchTarget(Message):
    """
    Patchable memory location descriptor.
    """

    patch_type: ParameterSpec
    """Data type at this address."""

    patch_offset: int
    """Memory address of the patch."""


@dataclass(eq=False, repr=False)
class RPCRequest(Message):
    """
    A single request object according to the JSONRPC standard.
    """

    method: str
    """The RPC function name."""

    params: Any
    """The RPC function arguments."""

    id: str
    """RPC request id (used to verify that request and response belong together)."""

    jsonrpc: str = "2.0"
    """The JSONRPC version."""

    client_timeout: Optional[float] = None
    """The client-side timeout for the request. The server itself may be configured with a timeout that is greater than the client-side timeout, in which case the server can choose to terminate any processing of the request."""

    client_key: Optional[str] = None
    """The ZeroMQ CURVE public key used to make the request, as received by the server. Empty if no key is used."""


@dataclass(eq=False, repr=False)
class RPCWarning(Message):
    """
    An individual warning emitted in the course of RPC processing.
    """

    body: str
    """The warning string."""

    kind: Optional[str] = None
    """The type of the warning raised."""


@dataclass(eq=False, repr=False)
class RPCReply(Message):
    """
    The reply for a JSONRPC request.
    """

    id: str
    """The RPC request id."""

    jsonrpc: str = "2.0"
    """The JSONRPC version."""

    result: Optional[Any] = None
    """The RPC result."""

    warnings: List[RPCWarning] = field(default_factory=list)
    """A list of warnings that occurred during request processing."""


@dataclass(eq=False, repr=False)
class RPCError(Message):
    """
    A error message for JSONRPC requests.
    """

    error: str
    """The error message."""

    id: str
    """The RPC request id."""

    jsonrpc: str = "2.0"
    """The JSONRPC version."""

    warnings: List[RPCWarning] = field(default_factory=list)
    """A list of warnings that occurred during request processing."""


@dataclass(eq=False, repr=False)
class TargetDevice(Message):
    """
    ISA and specs for a particular device.
    """

    isa: Dict[str, Dict]
    """Instruction-set architecture for this device."""

    specs: Dict[str, Dict]
    """Fidelities and coherence times for this device."""


@dataclass(eq=False, repr=False)
class RandomizedBenchmarkingRequest(Message):
    """
    RPC request payload for generating a randomized benchmarking sequence.
    """

    depth: int
    """Depth of the benchmarking sequence."""

    qubits: int
    """Number of qubits involved in the benchmarking sequence."""

    gateset: List[str]
    """List of Quil programs, each describing a Clifford."""

    seed: Optional[int] = None
    """PRNG seed. Set this to guarantee repeatable results."""

    interleaver: Optional[str] = None
    """Fixed Clifford, specified as a Quil string, to interleave through an RB sequence."""


@dataclass(eq=False, repr=False)
class RandomizedBenchmarkingResponse(Message):
    """
    RPC reply payload for a randomly generated benchmarking sequence.
    """

    sequence: List[List[int]]
    """List of Cliffords, each expressed as a list of generator indices."""


@dataclass(eq=False, repr=False)
class PauliTerm(Message):
    """
    Specification of a single Pauli term as a tensor product of Pauli factors.
    """

    indices: List[int]
    """Qubit indices onto which the factors of a Pauli term are applied."""

    symbols: List[str]
    """Ordered factors of a Pauli term."""


@dataclass(eq=False, repr=False)
class ConjugateByCliffordRequest(Message):
    """
    RPC request payload for conjugating a Pauli element by a Clifford element.
    """

    pauli: PauliTerm
    """Specification of a Pauli element."""

    clifford: str
    """Specification of a Clifford element."""


@dataclass(eq=False, repr=False)
class ConjugateByCliffordResponse(Message):
    """
    RPC reply payload for a Pauli element as conjugated by a Clifford element.
    """

    phase: int
    """Encoded global phase factor on the emitted Pauli."""

    pauli: str
    """Description of the encoded Pauli."""


@dataclass(eq=False, repr=False)
class NativeQuilRequest(Message):
    """
    Quil and the device metadata necessary for quilc.
    """

    quil: str
    """Arbitrary Quil to be sent to quilc."""

    target_device: TargetDevice
    """Specifications for the device to target with quilc."""


@dataclass(eq=False, repr=False)
class NativeQuilMetadata(Message):
    """
    Metadata for a native quil program.
    """

    final_rewiring: List[int] = field(default_factory=list)
    """Output qubit index relabeling due to SWAP insertion."""

    gate_depth: Optional[int] = None
    """Maximum number of successive gates in the native quil program."""

    gate_volume: Optional[int] = None
    """Total number of gates in the native quil program."""

    multiqubit_gate_depth: Optional[int] = None
    """Maximum number of successive two-qubit gates in the native quil program."""

    program_duration: Optional[float] = None
    """Rough estimate of native quil program length in nanoseconds."""

    program_fidelity: Optional[float] = None
    """Rough estimate of the fidelity of the full native quil program, uses specs."""

    topological_swaps: Optional[int] = None
    """Total number of SWAPs in the native quil program."""

    qpu_runtime_estimation: Optional[float] = None
    """The estimated runtime (milliseconds) on a Rigetti QPU for a protoquil program."""


@dataclass(eq=False, repr=False)
class NativeQuilResponse(Message):
    """
    Native Quil and associated metadata returned from quilc.
    """

    quil: str
    """Native Quil returned from quilc."""

    metadata: Optional[NativeQuilMetadata] = None
    """Metadata for the returned Native Quil."""


@dataclass(eq=False, repr=False)
class RewriteArithmeticRequest(Message):
    """
    A request type to handle compiling arithmetic out of gate parameters.
    """

    quil: str
    """Native Quil for which to rewrite arithmetic parameters."""


@dataclass(eq=False, repr=False)
class RewriteArithmeticResponse(Message):
    """
    The data needed to run programs with gate arithmetic on the hardware.
    """

    quil: str
    """Native Quil rewritten with no arithmetic in gate parameters."""

    original_memory_descriptors: Dict[str, ParameterSpec] = field(default_factory=dict)
    """The declared memory descriptors in the Quil of the related request."""

    recalculation_table: Dict[ParameterAref, str] = field(default_factory=dict)
    """A mapping from memory references to the original gate arithmetic."""


@dataclass(eq=False, repr=False)
class BinaryExecutableRequest(Message):
    """
    Native Quil and the information needed to create binary executables.
    """

    quil: str
    """Native Quil to be translated into an executable program."""

    num_shots: int
    """The number of times to repeat the program."""


@dataclass(eq=False, repr=False)
class BinaryExecutableResponse(Message):
    """
    Program to run on the QPU.
    """

    program: str
    """Execution settings and sequencer binaries."""

    memory_descriptors: Dict[str, ParameterSpec] = field(default_factory=dict)
    """Internal field for constructing patch tables."""

    ro_sources: List[Any] = field(default_factory=list)
    """Internal field for reshaping returned buffers."""


@dataclass(eq=False, repr=False)
class QuiltBinaryExecutableRequest(Message):
    """
    Native Quilt and the information needed to create binary executables.
    """

    quilt: str
    """Native Quilt to be translated into an executable program."""

    num_shots: int
    """The number of times to repeat the program."""


@dataclass(eq=False, repr=False)
class QuiltBinaryExecutableResponse(Message):
    """
    Program to run on the QPU.
    """

    program: str
    """Execution settings and sequencer binaries."""

    debug: Dict[str, Any]
    """Debug information associated with the translation process."""

    memory_descriptors: Dict[str, ParameterSpec] = field(default_factory=dict)
    """Internal field for constructing patch tables."""

    ro_sources: List[Any] = field(default_factory=list)
    """Internal field for reshaping returned buffers."""


@dataclass(eq=False, repr=False)
class PyQuilExecutableResponse(Message):
    """
    rpcQ-serializable form of a pyQuil Program object.
    """

    program: str
    """String representation of a Quil program."""

    attributes: Dict[str, Any]
    """Miscellaneous attributes to be unpacked onto the pyQuil Program object."""


@dataclass(eq=False, repr=False)
class QPURequest(Message):
    """
    Program and patch values to send to the QPU for execution.
    """

    program: Any
    """Execution settings and sequencer binaries."""

    patch_values: Dict[str, List[Any]]
    """Dictionary mapping data names to data values for patching the binary."""

    id: str
    """QPU request ID."""


@dataclass(eq=False, repr=False)
class QuiltCalibrationsRequest(Message):
    """
    A request for up-to-date Quilt calibrations.
    """

    target_device: TargetDevice
    """Specifications for the device to get calibrations for."""


@dataclass(eq=False, repr=False)
class QuiltCalibrationsResponse(Message):
    """
    Up-to-date Quilt calibrations.
    """

    quilt: str
    """Quilt code with definitions for frames, waveforms, and calibrations."""


@dataclass(eq=False, repr=False)
class GetExecutionResultsResponse(Message):
    """
    Results of a completed ExecutorJob execution.
    """

    buffers: Dict[str, Dict[str, Any]]
    """Result buffers for a completed ExecutorJob."""

    execution_duration_microseconds: int
    """Duration (in microseconds) ExecutorJob held exclusive access to quantum hardware."""



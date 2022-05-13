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

from rpcq.messages import *

@dataclass(eq=False, repr=False)
class Frame(Message):
    """
    A frame encapsulates any rotating frame
      relative to which control or readout waveforms may be defined.
    """

    direction: str
    """'rx' or 'tx'"""

    sample_rate: float
    """The sample rate [Hz] of the associated AWG/ADC"""

    frequency: float
    """The frame frequency [Hz]"""


@dataclass(eq=False, repr=False)
class Resources(Message):
    """
    The resources required by a job
    """

    qubits: List[str] = field(default_factory=list)
    """A list of qubits blocked/required by a job."""

    frames: Dict[str, Frame] = field(default_factory=dict)
    """RF/UHF frames by label."""

    frames_to_controls: Dict[str, str] = field(default_factory=dict)
    """Mapping of frames to control channels by labels."""


@dataclass(eq=False, repr=False)
class AbstractWaveform(Message):
    """
    A waveform envelope defined for a specific frame. This abstract class is made concrete by either a `Waveform` or a templated waveform such as `GaussianWaveform`
    """

    frame: str
    """The label of the associated tx-frame."""


@dataclass(eq=False, repr=False)
class Waveform(AbstractWaveform):
    """
    A waveform envelope defined by specific IQ values for a specific frame.
    """

    iqs: List[float] = field(default_factory=list)
    """The raw waveform envelope samples, alternating I and Q values."""


@dataclass(eq=False, repr=False)
class TemplateWaveform(AbstractWaveform):
    """
    A waveform envelope defined for a specific frame. A templated waveform is defined by a parameterized pulseshape rather than explicit IQ values. The message specification does not enforce that the duration is implementable on the hardware.
    """

    duration: float
    """Length of the pulse in seconds"""

    scale: float = 1.e+0
    """Scale to apply to waveform envelope"""

    phase: float = 0.0e+0
    """Phase [units of tau=2pi] to rotate the complex waveform envelope."""

    detuning: float = 0.0e+0
    """Modulation to apply to the waveform in Hz"""


@dataclass(eq=False, repr=False)
class GaussianWaveform(TemplateWaveform):
    """
    A Gaussian shaped waveform envelope defined for a specific frame.
    """

    fwhm: Optional[float] = None
    """Full Width Half Max shape paramter in seconds"""

    t0: Optional[float] = None
    """Center time coordinate of the shape in seconds. Defaults to mid-point of pulse."""


@dataclass(eq=False, repr=False)
class DragGaussianWaveform(TemplateWaveform):
    """
    A DRAG Gaussian shaped waveform envelope defined for a specific frame.
    """

    fwhm: Optional[float] = None
    """Full Width Half Max shape paramter in seconds"""

    t0: Optional[float] = None
    """Center time coordinate of the shape in seconds. Defaults to mid-point of pulse."""

    anh: float = -2.1e+8
    """Anharmonicity of the qubit, f01-f12 in (Hz)"""

    alpha: float = 0.0e+0
    """Dimensionless DRAG parameter"""


@dataclass(eq=False, repr=False)
class HermiteGaussianWaveform(TemplateWaveform):
    """
    Hermite-Gaussian shaped pulse. Reference: Effects of arbitrary laser
      or NMR pulse shapes on population inversion and coherence Warren S. Warren.
      81, (1984); doi: 10.1063/1.447644
    """

    fwhm: Optional[float] = None
    """Full Width Half Max shape paramter in seconds"""

    t0: float = 0.0e+0
    """Center time coordinate of the shape in seconds. Defaults to mid-point of pulse."""

    anh: float = -2.1e+8
    """Anharmonicity of the qubit, f01-f12 in Hz"""

    alpha: float = 0.0e+0
    """Dimensionless DRAG parameter"""

    second_order_hrm_coeff: float = 9.56e-1
    """Second order coefficient (see paper)"""


@dataclass(eq=False, repr=False)
class ErfSquareWaveform(TemplateWaveform):
    """
    Pulse with a flat top and rounded shoulders given by error functions
    """

    risetime: float = 1.e-9
    """The width of the rise and fall sections in seconds."""

    pad_left: float = 0.0e+0
    """Length of zero-amplitude padding before the pulse in seconds."""

    pad_right: float = 0.0e+0
    """Length of zero-amplitude padding after the pulse in seconds."""


@dataclass(eq=False, repr=False)
class FlatWaveform(TemplateWaveform):
    """
    Flat pulse.
    """

    iq: List[float] = field(default_factory=list)
    """Individual IQ point to hold constant"""


@dataclass(eq=False, repr=False)
class AbstractKernel(Message):
    """
    An integration kernel defined for a specific frame. This abstract class is made concrete by either a `FilterKernel` or `TemplateKernel`
    """

    frame: str
    """The label of the associated rx-frame."""


@dataclass(eq=False, repr=False)
class FilterKernel(AbstractKernel):
    """
    A filter kernel to produce scalar readout features from acquired readout waveforms.
    """

    iqs: List[float] = field(default_factory=list)
    """The raw kernel coefficients, alternating real and imaginary parts."""

    bias: float = 0.0e+0
    """The kernel is offset by this real value. Can be used to ensure the decision threshold lies at 0.0."""


@dataclass(eq=False, repr=False)
class TemplateKernel(AbstractKernel):
    """
    An integration kernel defined for a specific frame.
    """

    duration: float
    """Length of the boxcar kernel in seconds"""

    bias: float = 0.0e+0
    """The kernel is offset by this real value. Can be used to ensure the decision threshold lies at 0.0."""

    scale: float = 1.e+0
    """Scale to apply to boxcar kernel"""

    phase: float = 0.0e+0
    """Phase [units of tau=2pi] to rotate the kernel by."""

    detuning: float = 0.0e+0
    """Modulation to apply to the filter kernel in Hz"""


@dataclass(eq=False, repr=False)
class FlatKernel(TemplateKernel):
    """
    An unnormalized flat or boxcar integration kernel.
    """


@dataclass(eq=False, repr=False)
class BoxcarAveragerKernel(TemplateKernel):
    """
    A normalized flat or boxcar integration kernel.
    """


@dataclass(eq=False, repr=False)
class FilterNode(Message):
    """
    A node in the filter pipeline.
    """

    module: str
    """Absolute python module import path in which the filter
          class is defined."""

    filter_type: str
    """The type (class name) of the filter."""

    source: str
    """Filter node label of the input to this node."""

    publish: bool
    """If True, return the output of this node with the job
          results (and publish a stream for it)."""

    params: Dict[str, float] = field(default_factory=dict)
    """Additional filter parameters."""


@dataclass(eq=False, repr=False)
class DataAxis(Message):
    """
    A data axis allows to label element(s) of a stream.
    """

    name: str
    """Label for the axis, e.g., 'time' or 'shot_index'."""

    points: List[float] = field(default_factory=list)
    """The sequence of values along the axis."""


@dataclass(eq=False, repr=False)
class Receiver(Message):
    """
    The receiver settings generated by the low-level
      translator.
    """

    instrument: str
    """The instrument name"""

    channel: str
    """The instrument channel (label)"""

    stream: str
    """Name of the associated (raw) output stream that
          should be published."""

    publish: bool
    """Whether to publish the raw output stream."""

    data_axes: List[DataAxis] = field(default_factory=list)
    """Ordered list of DataAxis objects that together
          uniquely label each element in the stream."""


@dataclass(eq=False, repr=False)
class ParameterExpression(Message):
    """
    A parametric expression.
    """

    operator: str
    """The operator '+', '-', '*'. The operands can be
          constant floating point numbers or strings referencing a dynamic
          program parameter or a ParameterAref to index into an array or
          itself a ParameterExpression."""

    a: Any
    """The first operand"""

    b: Any
    """The second operand"""


@dataclass(eq=False, repr=False)
class Instruction(Message):
    """
    An instruction superclass.
    """

    time: float
    """The time at which the instruction is emitted [in seconds]."""


@dataclass(eq=False, repr=False)
class DebugMessage(Instruction):
    """
    Instructs the target to emit a specified debug message.
    """

    frame: str
    """The frame label that owns this debug message."""

    message: int
    """The 2-byte wide debug message to emit."""


@dataclass(eq=False, repr=False)
class Pulse(Instruction):
    """
    Instruction to play a pulse with some (modified) waveform
      envelope at a specific time on a specific frame.
    """

    frame: str
    """The tx-frame label on which the pulse is played."""

    waveform: str
    """The waveform label"""

    scale: Optional[float] = 1.e+0
    """Dimensionless (re-)scaling factor which is applied to
          the envelope."""

    phase: float = 0.0e+0
    """Static phase angle [units of tau=2pi] by which the
          envelope quadratures are rotated."""

    detuning: float = 0.0e+0
    """Detuning [Hz] with which the pulse envelope should be
          modulated relative to the frame frequency."""


@dataclass(eq=False, repr=False)
class FlatPulse(Instruction):
    """
    Instruction to play a pulse with a constant amplitude
      (except for phase modulation) at a specific time on a specific frame.
    """

    frame: str
    """The tx-frame label on which the pulse is played."""

    iq: List[float]
    """The I and Q value of the constant pulse."""

    duration: float
    """The duration of the pulse in [seconds], should be a
          multiple of the associated tx-frame's inverse sample rate."""

    phase: float = 0.0e+0
    """Static phase angle [units of tau=2pi] by which the
          envelope quadratures are rotated."""

    detuning: float = 0.0e+0
    """Detuning [Hz] with which the pulse envelope should be
          modulated relative to the frame frequency."""

    scale: Optional[float] = 1.e+0
    """Dimensionless (re-)scaling factor which is applied to
          the envelope."""


@dataclass(eq=False, repr=False)
class SetPhase(Instruction):
    """
    Set the phase of a frame to an absolute value at a specific
      time.
    """

    frame: str
    """The frame label for which to set the phase."""

    phase: float = 0.0e+0
    """Phase angle [units of tau=2pi] to update the frame phase
          to."""


@dataclass(eq=False, repr=False)
class ShiftPhase(Instruction):
    """
    Shift the phase of a frame by a relative value at a
      specific time.
    """

    frame: str
    """The frame label for which to set the phase."""

    delta: Any = None
    """Phase angle [units of tau=2pi] by which to shift the
          frame phase.  Can be a numerical value, a ParameterExpression or a
          ParameterAref."""


@dataclass(eq=False, repr=False)
class SwapPhases(Instruction):
    """
    Swap the phases of two tx-frames at a specific time.
    """

    frame_a: str
    """The first frame's label."""

    frame_b: str
    """The second frame's label."""


@dataclass(eq=False, repr=False)
class SetFrequency(Instruction):
    """
    Set the frequency of a tx-frame to a specific value at a
      specific time.
    """

    frame: str
    """The frame label for which to set the frequency."""

    frequency: float = 0.0e+0
    """The frequency [Hz] to set the frame frequency to."""


@dataclass(eq=False, repr=False)
class ShiftFrequency(Instruction):
    """
    Shift the frequency of a tx-frame by a specific amount at a
      specific time.
    """

    frame: str
    """The frame label for which to set the frequency."""

    delta: float = 0.0e+0
    """Frequency shift (new-old) [Hz] to apply to the frame
          frequency."""


@dataclass(eq=False, repr=False)
class SetScale(Instruction):
    """
    Set the scale of a tx-frame to a value at a specific time.
    """

    frame: str
    """The frame label for which to set the scale."""

    scale: float = 1.e+0
    """Scale (unitless) to apply to waveforms generated on the frame."""


@dataclass(eq=False, repr=False)
class Capture(Instruction):
    """
    Specify an acquisition on an rx-frame as well as the
      filters to apply.
    """

    frame: str
    """The rx-frame label on which to trigger the acquisition."""

    duration: float
    """The duration of the acquisition in [seconds]"""

    filters: List[str] = field(default_factory=list)
    """An ordered list of labels of filter kernels to apply to
          the captured waveform."""

    send_to_host: bool = True
    """Transmit the readout bit back to Lodgepole.
          (Unnecessary for fully calibrated active reset captures)."""

    phase: float = 0.0e+0
    """Static phase angle [units of tau=2pi] by which the
          envelope quadratures are rotated."""

    detuning: float = 0.0e+0
    """Detuning [Hz] with which the pulse envelope should be
          modulated relative to the frame frequency."""


@dataclass(eq=False, repr=False)
class MNIOConnection(Message):
    """
    Description of one side of an MNIO connection between two Tsunamis.
    """

    port: int
    """The physical Tsunami MNIO port, indexed from 0, 
          where this connection originates."""

    destination: str
    """The Tsunami where this connection terminates."""


@dataclass(eq=False, repr=False)
class Program(Message):
    """
    The dynamic aspects (waveforms, readout kernels, scheduled
  instructions and parameters) of a job.
    """

    waveforms: Dict[str, AbstractWaveform] = field(default_factory=dict)
    """The waveforms appearing in the program by waveform
          label."""

    filters: Dict[str, AbstractKernel] = field(default_factory=dict)
    """The readout filter kernels appearing in the program by
          feature label."""

    scheduled_instructions: List[Instruction] = field(default_factory=list)
    """The ordered sequence of scheduled instruction objects."""

    parameters: Dict[str, ParameterSpec] = field(default_factory=dict)
    """A mapping of dynamic parameter names to their type
          specification."""


@dataclass(eq=False, repr=False)
class ScheduleIRJob(Message):
    """
    The unit of work to be executed.
    """

    num_shots: int
    """How many repetitions the job should be executed for."""

    resources: Resources
    """The resources required by the job."""

    program: Program
    """The actual program to be executed."""

    operating_point: Dict[str, Dict] = field(default_factory=dict)
    """Operating points or static instrument channel settings
          (mapping control_name (instrument name) -> instrument channel settings
          (instrument settings) dictionary)."""

    filter_pipeline: Dict[str, FilterNode] = field(default_factory=dict)
    """The filter pipeline. Mapping of node labels to
          FilterNode's."""

    job_id: InitVar[Optional[str]] = None
    """A unique ID to help the submitter track the job."""

    def _extend_by_deprecated_fields(self, d):
        super()._extend_by_deprecated_fields(d)

    def __post_init__(self, job_id):
        if job_id is not None:
            warn('job_id is deprecated; please don\'t set it anymore')


@dataclass(eq=False, repr=False)
class RackMeta(Message):
    """
    Meta information about a rack configuration.
    """

    rack_id: Optional[str] = None
    """A unique identifier for the rack."""

    rack_version: Optional[int] = None
    """A version of the rack configuration."""

    schema_version: int = 0
    """A version of the rack configuration."""


@dataclass(eq=False, repr=False)
class QPU(Message):
    """
    Configuration info for the QPU
    """

    chip_label: str
    """The fabrication label for the QPU chip."""

    qubits: List[str] = field(default_factory=list)
    """A list of qubits labels."""

    controls: Dict[str, List] = field(default_factory=dict)
    """A mapping of control labels to tuples (instrument
          label, channel label)."""

    controls_by_qubit: Dict[str, List] = field(default_factory=dict)
    """A map of qubit label to list of controls that should be
          considered blocked when the qubit is part of a job execution."""


@dataclass(eq=False, repr=False)
class Instrument(Message):
    """
    Instrument settings.
    """

    address: str
    """The full address of a QPU."""

    module: str
    """Full python import path for the module that includes
          the instrument driver."""

    instrument_type: str
    """Instrument type (driver class name)."""

    mnio_connections: Dict[str, MNIOConnection] = field(default_factory=dict)
    """MNIO network connections between Tsunami instruments"""

    channels: Dict[str, Any] = field(default_factory=dict)
    """Mapping of channel labels to channel settings"""

    virtual: bool = False
    """Whether the instrument is virtual."""

    setup: Dict[str, Any] = field(default_factory=dict)
    """Any additional information used by the instrument for
          one-time-setup"""


@dataclass(eq=False, repr=False)
class DeployedRack(Message):
    """
    The rack configuration for lodgepole.
    """

    rack_meta: RackMeta
    """Meta information about the deployed rack."""

    qpu: QPU
    """Information about the QPU."""

    instruments: Dict[str, Instrument] = field(default_factory=dict)
    """Mapping of instrument name to instrument settings."""


@dataclass(eq=False, repr=False)
class AWGChannel(Message):
    """
    Configuration of a single RF channel.
    """

    sample_rate: float
    """The sampling rate [Hz] of the associated DAC/ADC
          component."""

    direction: str
    """'rx' or 'tx'"""

    lo_frequency: Optional[float] = None
    """The local oscillator frequency [Hz] of the channel."""

    gain: Optional[float] = None
    """If there is an amplifier, the amplifier gain [dB]."""

    delay: float = 0.0e+0
    """Delay [seconds] to account for inter-channel skew."""


@dataclass(eq=False, repr=False)
class QFDChannel(Message):
    """
    Configuration for a single QFD Channel.
    """

    channel_index: int
    """The channel index on the QFD, zero indexed from the
          lowest channel, as installed in the box."""

    direction: Optional[str] = "tx"
    """The QFD is a device that transmits pulses."""

    nco_frequency: Optional[float] = 0.0e+0
    """The DAC NCO frequency [Hz]."""

    gain: Optional[float] = 0.0e+0
    """The output gain on the DAC in [dB]. Note that this
          should be in the range -45dB to 0dB and is rounded to the
          nearest 3dB step."""

    delay: float = 0.0e+0
    """Delay [seconds] to account for inter-channel skew."""

    flux_current: Optional[float] = None
    """Flux current [Amps]."""

    relay_closed: Optional[bool] = None
    """Set the state of the Flux relay.
          True  - Relay closed, allows flux current to flow.
          False - Relay open, no flux current can flow."""


@dataclass(eq=False, repr=False)
class QGSChannel(Message):
    """
    Configuration for a single QGS Channel.
    """

    direction: Optional[str] = "tx"
    """The QGS is a device that transmits pulses."""

    nco_frequency: Optional[float] = 2.e+9
    """The DAC NCO frequency [Hz]."""

    gain: Optional[float] = 0.0e+0
    """The output gain on the DAC in [dB]. Note that this
          should be in the range -45dB to 0dB and is rounded to the
          nearest 3dB step."""

    channel_index: Optional[int] = 0
    """The channel index on the QGS, zero indexed from the lowest channel,
        as installed in the box."""

    delay: float = 0.0e+0
    """Delay [seconds] to account for inter-channel skew."""


@dataclass(eq=False, repr=False)
class QRTChannel(Message):
    """
    Configuration for a single QRT Channel.
    """

    direction: Optional[str] = "tx"
    """The QRT is a device that transmits readout pulses."""

    nco_frequency: Optional[float] = 1.25e+9
    """The DAC NCO frequency [Hz]."""

    gain: Optional[float] = 0.0e+0
    """The output gain on the DAC in [dB]. Note that this should be in the range
       -45dB to 0dB and is rounded to the nearest 3dB step."""

    channel_index: Optional[int] = 0
    """The channel index on the QRT, zero indexed from the lowest channel,
        as installed in the box."""

    delay: float = 0.0e+0
    """Delay [seconds] to account for inter-channel skew."""


@dataclass(eq=False, repr=False)
class QRRChannel(Message):
    """
    Configuration for a single QRR Channel.
    """

    channel_index: int
    """The channel index on the QRR, zero indexed from the lowest channel,
        as installed in the box."""

    direction: Optional[str] = "rx"
    """The QRR is a device that receives readout pulses."""

    nco_frequency: Optional[float] = 0.0e+0
    """The ADC NCO frequency [Hz]."""

    gain: Optional[float] = 0.0e+0
    """The input gain on the ADC in [dB]. Note that this should be in the range
       -45dB to 0dB and is rounded to the nearest 3dB step."""

    delay: float = 0.0e+0
    """Delay [seconds] to account for inter-channel skew."""


@dataclass(eq=False, repr=False)
class CWChannel(Message):
    """
    Configuration for a single CW Generator Channel.
    """

    channel_index: int = 0
    """The zero-indexed channel of the generator's output."""

    rf_output_frequency: Optional[int] = 1000000000
    """The CW generator's output frequency [Hz]."""

    rf_output_power: Optional[float] = 0.0e+0
    """The power of CW generator's output [dBm]."""

    rf_output_enabled: Optional[bool] = True
    """The state (on/off) of CW generator's output."""


@dataclass(eq=False, repr=False)
class QDOSlowFluxChannel(Message):
    """
    Configuration for a single QDO Slow Flux Channel.
    """

    channel_index: int
    """The channel index on the QDO, zero indexed from the
          lowest channel, as installed in the box. Flux index typically starts at 4."""

    flux_current: Optional[float] = None
    """Flux current [Amps]."""

    relay_closed: Optional[bool] = False
    """Set the state of the Flux relay.
          True  - Relay closed, allows flux current to flow.
          False - Relay open, no flux current can flow."""


@dataclass(eq=False, repr=False)
class QDOFastFluxChannel(Message):
    """
    Configuration for a single QDO Fast Flux Channel.
    """

    channel_index: int
    """The channel index on the QDO, zero indexed from the
          lowest channel, as installed in the box."""

    direction: Optional[str] = "tx"
    """The QDO is a device that transmits pulses."""

    delay: float = 0.0e+0
    """Delay [seconds] to account for inter-channel skew."""

    flux_current: Optional[float] = None
    """Flux current [Amps]."""


@dataclass(eq=False, repr=False)
class YokogawaGS200Channel(Message):
    """
    Configuration for a single Yokogawa GS200 Channel.
    """


@dataclass(eq=False, repr=False)
class LegacyUSRPSequencer(Message):
    """
    Configuration for a Legacy USRP Sequencer
    """

    tx_channel: Optional[str] = None
    """The label of the associated tx channel."""

    rx_channel: Optional[str] = None
    """The label of the associated rx channel."""


@dataclass(eq=False, repr=False)
class QFDSequencer(Message):
    """
    Configuration for a single QFD Sequencer.
    """

    tx_channel: str
    """The label of the associated channel."""

    sequencer_index: int
    """The sequencer index of this sequencer."""


@dataclass(eq=False, repr=False)
class QDOSequencer(Message):
    """
    Configuration for a single QDO Sequencer.
    """

    tx_channel: str
    """The label of the associated channel."""

    sequencer_index: int
    """The sequencer index of this sequencer."""


@dataclass(eq=False, repr=False)
class QFDx2Sequencer(Message):
    """
    Configuration for a single QFDx2 Sequencer.
    """

    tx_channel: str
    """The label of the associated channel."""

    sequencer_index: int
    """The sequencer index of this sequencer."""


@dataclass(eq=False, repr=False)
class QGSSequencer(Message):
    """
    Configuration for a single QGS Sequencer.
    """

    tx_channel: str
    """The label of the associated channel."""

    sequencer_index: int
    """The sequencer index of this sequencer."""


@dataclass(eq=False, repr=False)
class QGSx2Sequencer(Message):
    """
    Configuration for a single QGSx2 Sequencer.
    """

    tx_channel: str
    """The label of the associated channel."""

    sequencer_index: int
    """The sequencer index of this sequencer."""


@dataclass(eq=False, repr=False)
class QRTSequencer(Message):
    """
    Configuration for a single readout transmit (QRT) sequencer.
    """

    tx_channel: str
    """The label of the associated tx channel."""

    sequencer_index: int
    """The sequencer index (0-7) of this sequencer."""

    low_freq_range: Optional[bool] = False
    """Used to signal if this sequencer is in the low frequency configuration."""


@dataclass(eq=False, repr=False)
class QRTx2Sequencer(Message):
    """
    Configuration for a dual readout transmit (QRTx2) sequencer.
    """

    tx_channel: str
    """The label of the associated tx channel."""

    sequencer_index: int
    """The sequencer index (0-15) of this sequencer."""

    low_freq_range: Optional[bool] = False
    """Used to signal if this sequencer is in the low frequency configuration."""


@dataclass(eq=False, repr=False)
class QRRSequencer(Message):
    """
    Configuration for a single readout receive (QRR) sequencer.
    """

    rx_channel: str
    """The label of the associated rx channel."""

    sequencer_index: int
    """The sequencer index (0-15) to assign. Note that only
         sequencer 0 can return raw readout measurements."""


@dataclass(eq=False, repr=False)
class USICardSequencer(Message):
    """
    Configuration for the card which
      interfaces with the USI Target on the USRP.
    """

    tx_channel: str
    """The label of the associated channel."""


@dataclass(eq=False, repr=False)
class USITargetSequencer(Message):
    """
    Configuration for a single USITarget Sequencer.
    """

    tx_channel: str
    """The label of the associated intial tx channel."""

    rx_channel: str
    """The label of the associated initial rx channel."""

    sequencer_index: int
    """The sequencer index (0-7) to assign. Note that only
           sequencer 0 has the ability to use the NCO or capture raw readout
           streams."""


@dataclass(eq=False, repr=False)
class CWFrequencySweep(Message):
    """
    Configuration of a continuous wave frequency sweep.
    """

    start: float
    """Start frequency of the sweep, in Hz"""

    stop: float
    """Stop frequency of the sweep, in Hz"""

    num_pts: int
    """Number of frequency points to sample, cast to int."""

    source: int
    """Source port number"""

    measure: int
    """Measure port number"""


@dataclass(eq=False, repr=False)
class VNASettings(Message):
    """
    Configuration of VNA settings for a continuous wave sweep.
    """

    e_delay: float
    """Electrical delay in seconds from source to measure port"""

    phase_offset: float
    """Phase offset in degrees from measured to reported phase"""

    bandwidth: float
    """Bandwidth of the sweep, in Hz"""

    power: float
    """Source power in dBm"""

    freq_sweep: CWFrequencySweep
    """Frequency sweep settings"""

    averaging: int = 1
    """Sets the number of points to combine into an averaged
          trace"""


@dataclass(eq=False, repr=False)
class TimeBomb(Message):
    """
    Payload used to match a job with a particular execution
      target.
    """

    deadline: str
    """Deadline, specified in the format
          '%Y-%m-%dT%H:%M:%S.000Z', after which this job becomes unexecutable."""

    chip_label: str
    """Label string for the chip on which this job is meant to
          execute."""


@dataclass(eq=False, repr=False)
class MicrowaveSourceSettings(Message):
    """
    Configuration of Microwave Source settings for operating amplifiers.
    """

    frequency: float
    """Frequency setting for microwave source (Hz)."""

    power: float
    """Power setting for microwave source (dBm)."""

    output: bool
    """Output setting for microwave source. If true, the source will be turned on."""


@dataclass(eq=False, repr=False)
class ExecutorJob(Message):
    """
    Job which is sent directly to the executor
    """

    instrument_settings: Dict[str, Any]
    """Dict mapping instrument names to arbitrary instrument
          settings."""

    filter_pipeline: Dict[str, FilterNode]
    """The filter pipeline to process measured data."""

    receivers: Dict[str, Receiver]
    """Dict mapping stream names to receiver settings."""

    duration: Optional[float] = None
    """The total duration of the program execution in seconds."""

    timebomb: Optional[TimeBomb] = None
    """An optional payload used to match this job with a
          particular execution target."""


@dataclass(eq=False, repr=False)
class PatchableBinary(Message):
    """
    Tsunami binary with patching metadata for classical
      parameter modification.
    """

    base_binary: Any
    """Raw Tsunami binary object."""

    patch_table: Dict[str, PatchTarget]
    """Dictionary mapping patch names to their memory
          descriptors."""


@dataclass(eq=False, repr=False)
class ActiveReset(Message):
    """
    An active reset control sequence consisting of a repeated
      sequence of a measurement block and a feedback block conditional on the
      outcome of a specific measurement bit.  Regardless of the measurement
      outcomes the total duration of the control sequence is [attempts x
      (measurement_duration + feedback_duration)].  The total
      measurement_duration must be chosen to allow for enough time after any
      Capture commands for the measurement bit to propagate back to the gate
      cards that are actuating the feedback.
    """

    time: float
    """Time at which the ActiveReset begins in [seconds]."""

    measurement_duration: float
    """The duration of measurement block in [seconds]. The
          measurement bit is expected to have arrived on the QGS after
          this time relative to the overall start of the ActiveReset block."""

    feedback_duration: float
    """The duration of feedback block in [seconds]"""

    measurement_bit: int
    """The address of the readout bit to condition the
          feedback on.  The bit is first accessed after measurement_duration
          has elapsed."""

    attempts: int = 3
    """The number of times to repeat the active reset sequence."""

    measurement_instructions: List[Dict] = field(default_factory=list)
    """The ordered sequence of scheduled measurement
          instructions."""

    apply_feedback_when: bool = True
    """Apply the feedback when the measurement_bit equals the
          value of this flag."""

    feedback_instructions: List[Dict] = field(default_factory=list)
    """The ordered sequence of scheduled feedback instructions."""



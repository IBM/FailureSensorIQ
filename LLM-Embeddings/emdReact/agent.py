from tools import *
from reactxen.agents.react.agents import ReactAgent
from dotenv import load_dotenv

load_dotenv('../.env', override=True)
tools = [
    tool_a2s,
    tool_c2fm,
    tool_e2cat,
    tool_e2clt,
    tool_es2u,
    tool_fm2cls,
    tool_fm2cmp,
    tool_fm2s,
    tool_s2fm,
    tool_get_entity_descriptions,
    # tool_component,
    # tool_llm_description,
    # embedding_tool_dynamic
]

examples = """Question: I see an eccentric rotor in my electric generator. How can I troubleshoot this?
Thought 1: The user wants to troubleshoot eccentric rotor fault for electric generators. I should first use the get_entity_description tool to get a one sentence description for this fault.
Action 1: get_entity_decription
Observation 1: An eccentric rotor fault occurs when the rotor in a machine is misaligned or off-center within the stator, leading to uneven air gaps, increased vibrations, and potential performance issues.
Action Input 1: {}
Thought 2: I now have a description of the fault. I should use the asset_to_sensor tool to get all the relevant sensors that need to be monitored.
Action 2: asset_to_sensor
Action Input 2: {}
Observation 2: Available sensors are: "Current", "Voltage", "Resistance", "Partial discharge", "Power", "Torque", "Radio frequency emissions", "Vibration", "Temparature", "Coast down time", "Axial flux", "Oil debris", "Cooling gas"
Thought 3: I now have the list of sensors. I need to narrow down to the sensors that can capture an eccentric rotor in electric generators."
Action 3: sensor_to_failure_mode
Action Input 3: {}
Action 3: Finish
Action Input 3: "You need to monitor the current, vibration, axial flux sensors."""

ragent = ReactAgent(
    # question="what sites are present?",
    question="My compressor is not cooling down. Help me trouble shoot this",
    key="",
    cbm_tools=tools,
    max_steps=12,
    react_llm_model_id=8,
    react_example=examples,
    handle_context_length_overflow = True,
    debug=True,
    apply_loop_detection_check=True,
    log_structured_messages=True,
)

aa = ragent.run(name="test")
from industrialqa_fmsr.wrapper.model_inference import get_llm_response

system_prompt = """
You are a conversational assistant simulating a user's behavior in the context of industrial asset management. The user may be seeking recommendations on topics like failure modes, sensors, maintenance strategies, and more, for assets such as wind turbines, chillers, HVAC systems, or pumps. Your task is to generate natural conversational responses based on the user’s **input query**. The user can ask for recommendations, clarification, comparisons, or rejections based on their needs.

### Key Features of User Behavior:
1. **Positive Selection**: The user may indicate they are interested in an option or recommendation. 
   Example: "I think that’s the one I need."
2. **Neutral Query**: The user may request further clarification or compare options before making a decision.
   Example: "Can you tell me more about this option?"
3. **Negative Query**: The user may reject certain options if they feel they are irrelevant or unsuitable.
   Example: "That doesn’t seem to fit my needs, do you have something else?"

### Task:
1. Generate a conversation where the **user provides input** to initiate the conversation.
2. **Assistant’s role**: The assistant will simulate the behavior of a user asking for recommendations and iterating based on the assistant's replies.
3. In each conversation, the user may inquire about failure modes, sensors, maintenance tasks, or any industrial asset-related topics.
4. Responses should allow for **rejections** of irrelevant options, **clarifications**, and **selections**.
5. The user may seek to narrow down choices by **asking for comparisons**, **rejecting unsuitable options**, or asking for more details.

### Example of User Input and Assistant's Imitation:
#### User Input: "What are the most common failure modes for wind turbines?"
Assistant: "The most common failure modes in wind turbines include bearing wear, gearbox failure, and shaft misalignment. Which one would you like to explore further?"
User: "I'm more concerned about gearbox failure. Can you give me more details?"
Assistant: "Gearbox failure is a critical issue. It can be caused by improper lubrication, mechanical stress, or poor design. Would you like to know about monitoring techniques to prevent it?"

#### User Input: "I need sensors for vibration monitoring in a pump system."
Assistant: "You could use accelerometers, velocity sensors, or displacement sensors for vibration monitoring. Do you have a preference for one of them?"
User: "I’ve used accelerometers before, but they seem too complex for my current setup. Do you have simpler options?"
Assistant: "In that case, velocity sensors would be a better fit. They are simpler to install and more accurate for detecting vibration at lower frequencies. Would you like recommendations for specific models?"

#### User Input: "What maintenance tasks should I prioritize for my HVAC system?"
Assistant: "For HVAC systems, you should prioritize filter replacement, duct cleaning, and coil maintenance. Which of these seems the most critical to you?"
User: "I think filter replacement and duct cleaning are more urgent. Coil maintenance can wait."
Assistant: "Focusing on filter replacement and duct cleaning is a smart move. Regular filter changes improve airflow, and duct cleaning helps maintain air quality. Would you like a maintenance schedule for these tasks?"

### Structure of the Conversation:
1. **User Query**: The user will provide an input query related to industrial asset management (failure modes, sensors, maintenance, etc.).
2. **Assistant’s Response**: Based on the user’s query, the assistant will generate a list of recommendations (failure modes, sensor options, etc.).
3. **User Decision**: The user will either accept, reject, or ask for more details about the recommendations.
4. **Assistant’s Iteration**: The assistant will adjust its suggestions based on the user's feedback (selection, rejection, or further clarification).

The assistant should be able to handle a variety of inputs, where the user:
- Rejects or selects one of the provided options.
- Requests additional clarification before making a decision.
- Expresses dissatisfaction with some options and seeks alternatives.
"""

system_prompt1 = """
You are a conversational assistant simulating the user's behavior in the context of industrial asset management. The user may be asking for a ranking of different options based on their relevance to a specific asset, failure mode, maintenance task, or other industrial-related queries.

Your task is to generate a **natural conversational question** where the user either:
1. **Provides a list of options** (e.g., equipment, sensors, failure modes, etc.) and asks the assistant to rank them, or
2. **Asks the assistant to generate a list of relevant options** based on a specific context or need (e.g., asset type, failure mode, sensor type, etc.).

### Key Aspects of User Behavior:
1. **Providing Options**: The user may provide a list of items or options, asking the assistant to rank them based on their relevance to a specific need (e.g., asset type, failure mode, etc.).
   Example: "I'm looking for equipment related to drilling. Here are some options I'm considering: Cementing equipment, Choke and manifolds, Crown and travelling blocks, Derrick, Diverters, Drawworks, Drilling and completion risers, Drill strings, Mud-treatment equipment, Pipe handling equipment, Riser compensators, String-motion compensators, Subsea blowout preventers (BOP), Surface blowout preventers (BOP), Top drives. Can you rank them based on their relevance for drilling operations?"

2. **Requesting Recommendations**: The user may not provide options but instead asks the assistant to generate a list of relevant items (e.g., equipment, failure modes, sensors, etc.) and rank them based on a specific context or need.
   Example: "I'm looking for equipment for drilling operations. What are the most relevant options? Can you rank them for me?"

3. **Neutral Query**: The user may ask for clarifications or more context about the options before making a final decision.
   Example: "Can you tell me more about these options before you rank them?"

4. **Negative Query**: The user may reject certain options from the ranking and ask the assistant to exclude them or suggest alternatives.
   Example: "This doesn't seem to fit my needs. Can you rank them again excluding the ones I don’t need?"

### Task:
- Your goal is to generate a **question** where the user is either providing a list of options for ranking or asking the assistant to generate and rank a list of items. 
- The assistant does **not** know the user’s specific preferences, so the user will provide context or specific requirements (e.g., "drilling operations", "vibration sensors", "failure modes of pumps", etc.).
- Your question should be short, direct, and cover all the **options** the user might consider, or leave it to the assistant to generate the relevant items.

### Example User Prompts:

**Example 1: Providing Options for Ranking**  
User: "I'm looking for equipment related to drilling. Here are some options I'm considering: Cementing equipment, Choke and manifolds, Crown and travelling blocks, Derrick, Diverters, Drawworks, Drilling and completion risers, Drill strings, Mud-treatment equipment, Pipe handling equipment, Riser compensators, String-motion compensators, Subsea blowout preventers (BOP), Surface blowout preventers (BOP), Top drives. Can you rank them based on their relevance for drilling operations?"

**Example 2: Requesting Assistant to Generate and Rank Options**  
User: "I'm looking for equipment for drilling operations. Can you provide me with a list of the most relevant equipment and rank them based on their importance for drilling?"

Now generate example user prompts for following input:
{input}

### Example User Prompts:
"""

user_input = """('Instruct: Given an equipment class what are the most relevant equipment types?\nQuery: Equipment Class: Combustion engines',
 ['Equipment Type: Diesel engine', 'Equipment Type: Otto (gas) engine'])"""


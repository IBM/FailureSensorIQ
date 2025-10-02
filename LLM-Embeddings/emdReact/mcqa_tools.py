from langchain.tools import Tool
import json
from utils import get_llm_response
from sentence_transformers import models, SentenceTransformer
from accelerate import Accelerator
class DynamicEmbeddingSearchTool:
    def __init__(self, ckpt_path):
        # Initialize the embedding model and LLM
        transformer = models.Transformer(ckpt_path)
        pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="lasttoken")
        normalize = models.Normalize()
        device = Accelerator().device
        self.embedding_model = SentenceTransformer(modules=[transformer, pooling, normalize], device=device, trust_remote_code=True)
        
    def search(self, query, instruction, documents, top_k=1):
        # Step 1: Convert the comma-separated string of documents into a list
        # documents = documents_str.split(',')
        prompt = f"Instruction: {instruction}\nQuery: {query}"
        q_emb = self.embedding_model.encode(prompt)
        docs_emb = self.embedding_model.encode(documents)
        sim = self.embedding_model.similarity(q_emb, docs_emb)
        return documents[sim.argmax().item()]

# Wrap it as a LangChain Tool
# embedding_tool_dynamic = Tool(
#     name="dynamic_embedding_search_tool",
#     func=dynamic_embedding_search_tool,
#     description="Takes a query, instruction, and comma-separated documents at runtime. It performs an embedding-based search to return relevant documents."
# )

def get_tool_data(key, fname='../data/industrial/processed/all_items.json'):
    fname = '../data/industrial/processed/all_items.json'
    with open(fname, 'r') as f:
        data = json.load(f)
        return data[key]

# ckpt_path = f'../models/google-bert_bert-base-uncased_fmsr_ft_bs32_poolmean_pdesc0.4_positiveratio_0.5seed_42/checkpoint-800'
ckpt_path = '../models/checkpoint-4400'
tool = DynamicEmbeddingSearchTool(ckpt_path)

def a2s(asset: str):
    instruction = 'For a given asset and asset class what sensors are relevant?'
    query = f'Asset: {asset}'
    documents = get_tool_data('sensors')
    # tool = DynamicEmbeddingSearchTool()
    return tool.search(query, instruction, documents)

def c2fm(component: str):
    instruction = 'Given an asset component, what are possible failure modes it can experience?'
    query = f'Component: {component}'
    documents = get_tool_data('faults')
    # tool = DynamicEmbeddingSearchTool()
    return tool.search(query, instruction, documents)

def e2cat(equipment: str):
    instruction = 'Given an equipment category what are the most relevant equipments?'
    query = f'Equipment Category: {equipment}'
    documents = get_tool_data('eq_class')
    # tool = DynamicEmbeddingSearchTool()
    return tool.search(query, instruction, documents)

def e2clt(equipment: str):
    instruction = 'Given an equipment class what are the most relevant equipment types?'
    query = f'Equipment Class: {equipment}'
    documents = get_tool_data('eq_type')
    # tool = DynamicEmbeddingSearchTool()
    return tool.search(query, instruction, documents)

def es2u(unit: str):
    instruction = 'For a given equipment unit what are the component names and their groups that it consists of?'
    query = f'Equipment unit: {unit}'
    documents = get_tool_data('eq_subunits')
    # tool = DynamicEmbeddingSearchTool()
    return tool.search(query, instruction, documents)

def fm2cls(failure_mode: str):
    instruction = 'For a given equipment unit what are the component names and their groups that it consists of?'
    query = f'Equipment unit: {failure_mode}'
    documents = get_tool_data('eq_subunits')
    # tool = DynamicEmbeddingSearchTool()
    return tool.search(query, instruction, documents)

def fm2cmp(failure_mode: str):
    instruction = 'Given a failure description what is its failure class?'
    query = f'Failure Description: {failure_mode}'
    documents = get_tool_data('components')
    # tool = DynamicEmbeddingSearchTool()
    return tool.search(query, instruction, documents)

def fm2s(asset: str, failure_mode: str, asset_category: str, asset_description: str, fault_description: str, candidate_sensors: list):
    """
    For a given asset, its category, and a sensor returns the faults that can be detected 
    asset (str): The industrial asset name
    failure_mode (str): The asset's failure mode (fault)
    asset_category (str): The asset's category with values: electrical, mechanical, rotating
    asset_description (str): A one sentence description for the asset
    fault_description (str): A one sentence description for the failure mode (fault)
    candidate_sensors (list): The candidate failure modes to narrow down
    Returns:
    (list) the most probable sensors that can detect the failure mode
    """
    instruction = 'For a given asset, its category, and a fault, what sensors can detect it?'
    query = f'Asset: {asset}, Category: {asset_category}, Fault: {failure_mode}, Asset Description: {asset_description}, Fault Description: {fault_description}'
    # documents = get_tool_data('sensors')
    # tool = DynamicEmbeddingSearchTool()
    candidate_sensors = ['Sensor: ' + sensor for sensor in candidate_sensors]
    retrieved = tool.search(query, instruction, candidate_sensors)
    return retrieved.replace('Sensor: ', '')

def s2fm(asset: str, sensor: str, asset_category: str, asset_description: str, sensor_description: str, candidate_failure_modes: list):
    """
    For a given asset, its category, and a sensor returns the failure modes that can be detected 
    asset (str): The industrial asset name
    sensor (str): The asset's sensor
    asset_category (str): The asset's category with values: electrical, mechanical, rotating
    asset_description (str): A one sentence description for the asset
    sensor_description (str): A one sentence description for the sensor
    candidate_failure_modes (list): The candidate failure modes to narrow down
    Returns:
    (list) the most probable failure modes that the sensor can detect
    """
    instruction = 'For a given asset, its category, and a sensor what are the faults that can be detected?'
    query = f'Asset: {asset}, Asset Description: {asset_description}, Category: {asset_category}, Sensor: {sensor}, Sensor Description: {sensor_description}'
    # documents = get_tool_data('faults')
    # tool = DynamicEmbeddingSearchTool()
    candidate_failure_modes = ['Failure Mode: ' + item for item in candidate_failure_modes]
    retrieved = tool.search(query, instruction, candidate_failure_modes)
    return retrieved.replace('Failure Mode: ', '')

# we can have a tool for calling for creating a one sentence descriptions
def get_entity_descriptions(entity: str):
    prompt = f'Provide me a one sentence description for {entity}.'
    response = get_llm_response('meta-llama/llama-3-3-70b-instruct', prompt, None)
    return response


# Create a LangChain Tool using the dynamic embedding search
# def dynamic_embedding_search_tool(query, instruction, documents_str):
#     tool = DynamicEmbeddingSearchTool()
#     return tool.search(query, instruction, documents_str)

# Wrapping each function with LangChain's `Tool` class
tool_a2s = Tool(
    name="asset_to_sensor",
    func=a2s,
    description="Returns a comma separated list of the sensors related to a given asset."
)

tool_c2fm = Tool(
    name="component_to_failure_modes",
    func=c2fm,
    description="Returns a comma separated list of the possible failure modes for the given component."
)

tool_e2cat = Tool(
    name="equipment_category_to_class_type",
    func=e2cat,
    description="Returns a comma separated list with the most relevant the category corresponding to a given equipment class type."
)

tool_e2clt = Tool(
    name="equipment_class_type_to_category",
    func=e2clt,
    description="Returns a comma separated list with the most relevant class types that fall under a given equipment category."
)

tool_es2u = Tool(
    name="equipment_unit_to_subunit",
    func=es2u,
    description="Returns the unit that a given subunit belongs to."
)

tool_fm2cls = Tool(
    name="failure_mode_to_class",
    func=fm2cls,
    description="Returns the class for a given failure mode as a string."
)

tool_fm2cmp = Tool(
    name="failure_mode_to_component",
    func=fm2cmp,
    description="Returns a comma separated list of the components that can experience the given failure mode."
)

tool_fm2s = Tool(
    name="failure_mode_to_sensor",
    func=fm2s,
    description="Returns a comma separated list of the sensors that can capture the given failure mode."
)

tool_s2fm = Tool(
    name="sensor_to_failure_mode",
    func=s2fm,
    description="Returns a comma separated list of the failure modes that can detect the given sensor."
)
tool_get_entity_descriptions = Tool(
    name="get_entity_descriptions",
    func=get_entity_descriptions,
    description="Returns a one sentence description of the given entity as string."
)

# tool_component = Tool(
#     name="get_component",
#     func=get_component,
#     description="Returns the count and a list of all components."
# )



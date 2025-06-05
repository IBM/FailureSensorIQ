import subprocess

def run_uq_benchmark(model_name, prompt_type='chat', dataset='full'):
    command = f'cd llm_uncertainty_bench && ./run.sh {model_name} {prompt_type} {dataset}'
    result = subprocess.run(
        command,
        capture_output = True, # Python >= 3.7 only
        text = True, # Python >= 3.7 only
        shell=True
    )
    result_dict = {}
    print(result.stdout)
    for line in result.stdout.split('\n'):
        if ':' in line:
            k, v = [item.lower().strip().replace(' ', '_') for item in line.split(':')]
            v = float(v) / 100
            result_dict[k] = v
    return result_dict
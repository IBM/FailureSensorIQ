import subprocess

def run_uq_benchmark(model_name):
    command = f'cd llm_uncertainty_bench && ./run.sh {model_name}'
    result = subprocess.run(
        command,
        capture_output = True, # Python >= 3.7 only
        text = True, # Python >= 3.7 only
        shell=True
    )
    keyword = 'Average UAcc'
    for line in result.stdout.split('\n'):
        if keyword in line:
            uq_score = float(line.split(':')[-1])
            break
    return uq_score
curl -X 'POST' \
  'https://restricted-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/deepseek-r1/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'RITS_API_KEY: 6e32ba2967a7153036d53126cd5bfdca' \
  -H 'Content-Type: application/json' \
  -d '{"model": "deepseek-ai/deepseek-r1",
    "messages": [
      {
        "role": "user",
        "content": "Write a haiku about recursion in programming."
      },
      {
        "role": "assistant",
        "content": "You are a helpful assistant."
      }
    ]
   }'
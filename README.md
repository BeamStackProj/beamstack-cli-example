# demo

# Running the inference

```sh
python main.py --es_client_args '{"hosts": ["https://localhost:9200"], "basic_auth": ["elastic", ""], "verify_certs": false}' --es_store_args '{"index_name": "beamstack", "text_field": "documentation", "vector_field": "documentation_vector"}'
```
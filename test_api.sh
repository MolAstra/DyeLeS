curl -X 'POST' \
  'http://127.0.0.1:9000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "smiles": "CC(=O)OC1=CC=CC=C1C(O)=O"
}'
  
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "smiles": "N#Cc1cc2ccc(O)cc2oc1=O"
}'
  
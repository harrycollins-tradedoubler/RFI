# RFI Agent Frontend

Public test frontend for your n8n workflow:
- Workflow name: `RFI Agent`
- Workflow ID: `OxplTFYVkEM_iGJMuU5RX`
- Webhook URL: `https://coe-n8n.coe-untrust-eu-de.prod.tddrift.net/webhook/6b55959d-96d7-46c5-be2e-11fa3736e1c3`

The UI reuses the Agent Hub style (same blue palette and layout language) but is focused on one testable RFI chat agent.

## Input modes

- `Chat`: ask a single question or paste multiple questions in one message (one per line).
- `File Upload`: upload a questionnaire and answer all parsed questions in one run.
  - Supported files: `.txt`, `.md`, `.csv`, `.json`, `.docx`, `.xlsx`, `.xls`
  - Limit: first 25 unique parsed questions

## Run locally

```bash
npm install
npm run dev
```

## Deploy publicly on Vercel

1. Push this `rfi-agent-frontend` folder to a Git repository.
2. In Vercel, import the repo and set the project root to `rfi-agent-frontend`.
3. Add environment variables in Vercel:
   - `VITE_RFI_DIRECT_WEBHOOK_URL=https://coe-n8n.coe-untrust-eu-de.prod.tddrift.net/webhook/6b55959d-96d7-46c5-be2e-11fa3736e1c3`
4. Deploy. Vercel will provide a public URL your colleague can use.

## Notes

- The app calls the n8n webhook directly from the browser.
- n8n must allow CORS for `https://rfi-agent-frontend.vercel.app` (and any other frontend origins you use).
- If replies fail, verify the n8n workflow is active and the webhook URL is reachable from your browser/VPN.

"""Download the SAGE dataset from Synapse into project-root/data.

Needs a Synapse personal access token in the environment variable
SYNAPSE_AUTH_TOKEN (see .env.example). The token is never printed or stored.
"""
import os
import synapseclient
import synapseutils

if not os.environ.get("SYNAPSE_AUTH_TOKEN", "").strip():
    raise SystemExit(
        "SYNAPSE_AUTH_TOKEN is not set. Create a Synapse personal access token "
        "(view + download scope) and export it before running:\n"
        "    export SYNAPSE_AUTH_TOKEN=<your token>"
    )

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)

syn = synapseclient.Synapse()
syn.login(authToken=os.environ["SYNAPSE_AUTH_TOKEN"].strip())

files = synapseutils.syncFromSynapse(syn, 'syn75397327', path=data_dir)

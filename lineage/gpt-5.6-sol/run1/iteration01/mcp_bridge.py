"""Small streamable-HTTP MCP bridge executed through the task container.

The kernel deliberately exposes only a shell execution primitive.  MCP servers in
scenario tasks are reachable from that container, so this module bootstraps them
there and presents their advertised functions as normal model tools.
"""
import base64
import json
import re


_SERVER_RE = re.compile(
    r"^-\s+([^:\n]+):\s+streamable-http\s+at\s+(https?://\S+)\s*$", re.M
)


def declared_servers(instruction):
    return [(name.strip(), url.strip()) for name, url in _SERVER_RE.findall(instruction)]


def _b64(value):
    return base64.b64encode(value.encode()).decode()


def _script(body):
    # Quoted heredocs prevent task/user strings from becoming shell syntax.
    return "python3 - <<'PY'\n" + body + "\nPY"


async def connect(world, name, url):
    """Initialize one server and return (session id, advertised tools)."""
    code = f'''import urllib.request,json,base64
url=base64.b64decode("{_b64(url)}").decode()
def post(obj,sid=None):
 h={{"Content-Type":"application/json","Accept":"application/json, text/event-stream"}}
 if sid: h["Mcp-Session-Id"]=sid
 q=urllib.request.Request(url,data=json.dumps(obj).encode(),headers=h)
 with urllib.request.urlopen(q,timeout=45) as r:
  raw=r.read().decode(errors="replace"); heads=dict(r.headers)
 vals=[]
 for line in raw.splitlines():
  if line.startswith("data:"):
   try: vals.append(json.loads(line[5:].strip()))
   except Exception: pass
 return heads,vals
h,v=post({{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"2025-03-26","capabilities":{{}},"clientInfo":{{"name":"agent-harness","version":"1"}}}}}})
sid=h.get("mcp-session-id") or h.get("Mcp-Session-Id")
post({{"jsonrpc":"2.0","method":"notifications/initialized","params":{{}}}},sid)
h,v=post({{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{{}}}},sid)
answer=next((x for x in v if x.get("id")==2),v[-1] if v else {{}})
out={{"sid":sid,"tools":answer.get("result",{{}}).get("tools",[])}}
print("MCP_BRIDGE:"+base64.b64encode(json.dumps(out).encode()).decode())'''
    result = await world.exec(_script(code), timeout_s=60)
    marker = next((x[11:] for x in result.stdout.splitlines() if x.startswith("MCP_BRIDGE:")), None)
    if result.return_code or not marker:
        raise RuntimeError(f"could not connect to MCP server {name}: {(result.stderr or result.stdout)[-1000:]}")
    data = json.loads(base64.b64decode(marker).decode())
    if not data.get("sid"):
        raise RuntimeError(f"MCP server {name} supplied no session id")
    return data["sid"], data.get("tools", [])


async def call(world, url, sid, tool_name, arguments):
    payload = json.dumps({
        "jsonrpc": "2.0", "id": 7, "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    })
    code = f'''import urllib.request,json,base64
url=base64.b64decode("{_b64(url)}").decode()
sid=base64.b64decode("{_b64(sid)}").decode()
payload=base64.b64decode("{_b64(payload)}")
h={{"Content-Type":"application/json","Accept":"application/json, text/event-stream","Mcp-Session-Id":sid}}
q=urllib.request.Request(url,data=payload,headers=h)
with urllib.request.urlopen(q,timeout=900) as r: raw=r.read().decode(errors="replace")
vals=[]
for line in raw.splitlines():
 if line.startswith("data:"):
  try: vals.append(json.loads(line[5:].strip()))
  except Exception: pass
answer=next((x for x in vals if x.get("id")==7),vals[-1] if vals else {{}})
print("MCP_BRIDGE:"+base64.b64encode(json.dumps(answer).encode()).decode())'''
    result = await world.exec(_script(code), timeout_s=930)
    marker = next((x[11:] for x in result.stdout.splitlines() if x.startswith("MCP_BRIDGE:")), None)
    if result.return_code or not marker:
        return f"MCP transport error: {(result.stderr or result.stdout)[-2000:]}"
    response = json.loads(base64.b64decode(marker).decode())
    if response.get("error"):
        return "MCP error: " + json.dumps(response["error"], ensure_ascii=False)
    value = response.get("result", {})
    pieces = []
    for item in value.get("content", []):
        if item.get("type") == "text":
            pieces.append(item.get("text", ""))
        else:
            pieces.append(json.dumps(item, ensure_ascii=False))
    if not pieces and "structuredContent" in value:
        pieces.append(json.dumps(value["structuredContent"], ensure_ascii=False))
    text = "\n".join(pieces) or json.dumps(value, ensure_ascii=False)
    if value.get("isError"):
        text = "MCP tool reported an error:\n" + text
    return text

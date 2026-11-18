"""Optional loopback-only knowledge service and read-only HTTP tool adapter."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
from urllib.parse import urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler, ProxyHandler
from knowledge import KnowledgeBase


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('知识API不允许重定向')


class KnowledgeAPIClient:
    def __init__(self, endpoint):
        parts = urlparse(endpoint)
        if (parts.scheme != 'http' or parts.hostname not in {'127.0.0.1', 'localhost', '::1'}
                or parts.username or parts.password or parts.query or parts.fragment or parts.path not in {'', '/'}):
            raise ValueError('演示知识API仅允许本机HTTP地址，如http://127.0.0.1:8765')
        self.endpoint = endpoint.rstrip('/')
        self.version = None

    def search(self, query, top_k=3):
        if not isinstance(query,str) or not 1 <= len(query) <= 2000 or type(top_k) is not int or not 1 <= top_k <= 5:
            raise ValueError('检索参数无效')
        request = Request(self.endpoint+'/search',
            data=json.dumps({'query':query,'top_k':top_k}).encode(),
            headers={'Content-Type':'application/json'},method='POST')
        with build_opener(ProxyHandler({}), NoRedirect).open(request,timeout=5) as response:
            raw=response.read(200_001)
        if len(raw)>200_000: raise ValueError('知识API返回过大')
        value=json.loads(raw)
        if not isinstance(value,dict) or not isinstance(value.get('hits'),list) or not isinstance(value.get('version'),str):
            raise ValueError('知识API响应无效')
        seen = set()
        if len(value['hits']) > 5:
            raise ValueError('知识API条目数无效')
        for h in value['hits']:
            if not isinstance(h,dict) or any(not isinstance(h.get(k),str) or not h[k].strip() for k in ('id','title','source','text')):
                raise ValueError('知识API条目无效')
            if (not re.fullmatch(r'[A-Za-z0-9_-]{1,60}', h['id']) or h['id'] in seen
                    or len(h['text']) > 12000 or type(h.get('demo', True)) is not bool):
                raise ValueError('知识API编号、长度或示例标识无效')
            seen.add(h['id'])
            h.setdefault('demo', True)
        self.version=value['version']
        return value['hits'][:top_k]


def create_server(port=8765, kb_path=None):
    kb=KnowledgeBase(kb_path)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass  # Avoid logging task text.
        def do_POST(self):
            try:
                if self.path!='/search': raise ValueError('not found')
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=10000: raise ValueError('invalid length')
                value=json.loads(self.rfile.read(size))
                query=value.get('query');k=value.get('top_k',3)
                if not isinstance(query,str) or not 1<=len(query)<=2000 or type(k) is not int or not 1<=k<=5:
                    raise ValueError('invalid input')
                raw=json.dumps({'hits':kb.search(query,k),'version':kb.version},ensure_ascii=False).encode()
                self.send_response(200)
            except (ValueError,TypeError,AttributeError):
                raw=b'{"error":"invalid request"}'
                self.send_response(400)
            self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(raw)))
            self.end_headers();self.wfile.write(raw)
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765)
    p.add_argument('--knowledge',default=None);args=p.parse_args()
    server=create_server(args.port,args.knowledge)
    print(f'只读知识API：http://127.0.0.1:{server.server_port}/search')
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()

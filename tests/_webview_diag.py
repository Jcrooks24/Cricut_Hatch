"""Debug harness: open the real Hatch page, report whether the JS initialized,
then auto-close so nothing hangs. Prints findings to stdout."""
import os, sys, time, threading, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import webview
import hatch_app

def on_loaded():
    try:
        probe = ("(function(){try{return JSON.stringify({"
                 "hasS: typeof S!=='undefined',"
                 "step: (typeof S!=='undefined'? S.step : null),"
                 "nextBtn: !!document.getElementById('nextBtn'),"
                 "genBtn: !!document.getElementById('genBtn'),"
                 "apiReady: !!(window.pywebview&&window.pywebview.api),"
                 "err: window.__diagErr||null"
                 "});}catch(e){return 'PROBE_ERR '+e;}})()")
        print("PAGE LOADED ->", win.evaluate_js(probe), flush=True)
    except Exception as e:
        print("evaluate_js failed:", e, flush=True)

def watchdog():
    time.sleep(7)
    print("watchdog: destroying window", flush=True)
    try: win.destroy()
    except Exception as e: print("destroy err:", e, flush=True)

api = hatch_app.Api()
# inject a global error trap at the very top of the page so JS load errors surface
html = hatch_app._page_html().replace(
    "<body>", "<body><script>window.onerror=function(m,s,l,c){window.__diagErr=m+' @'+l+':'+c;};</script>", 1)
win = webview.create_window("Hatch diag", html=html, js_api=api)
api.window = win
win.events.loaded += on_loaded
threading.Thread(target=watchdog, daemon=True).start()
print("starting webview (will auto-close in 7s)...", flush=True)
try:
    webview.start(debug=False)
    print("webview ended cleanly", flush=True)
except Exception:
    traceback.print_exc()

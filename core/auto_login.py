"""
core/auto_login.py — Fully automatic Zerodha login using TOTP.
Writes token.json so MarketHub.load_kite() works unchanged.
"""
import json, logging, os, sys, time
from datetime import date
import pyotp, requests
from kiteconnect import KiteConnect

log = logging.getLogger("core.auto_login")
TOKEN_FILE = "token.json"

def _save_token(api_key, access_token):
    with open(TOKEN_FILE, "w") as f:
        json.dump({"api_key": api_key, "access_token": access_token,
                   "date": str(date.today())}, f, indent=2)
    log.info("token.json saved.")

def _reuse_today(kite):
    try:
        data = json.load(open(TOKEN_FILE))
        if data.get("date") != str(date.today()):
            return False
        kite.set_access_token(data["access_token"])
        kite.profile()           # verify still valid
        log.info("Reusing today's token.")
        return True
    except:
        return False

def auto_login() -> KiteConnect:
    api_key     = os.environ["ZERODHA_API_KEY"].strip()
    api_secret  = os.environ["ZERODHA_API_SECRET"].strip()
    user_id     = os.environ["ZERODHA_USER_ID"].strip()
    password    = os.environ["ZERODHA_PASSWORD"].strip()
    totp_secret = os.environ["ZERODHA_TOTP_SECRET"].strip()

    kite = KiteConnect(api_key=api_key)
    if _reuse_today(kite):
        return kite

    log.info("Fresh Zerodha auto-login starting...")
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})

    # Step 1 — credentials
    r = s.post("https://kite.zerodha.com/api/login",
               data={"user_id": user_id, "password": password}, timeout=15)
    r.raise_for_status()
    body = r.json()
    if body.get("status") != "success":
        log.error(f"Login failed: {body.get('message')}"); sys.exit(1)
    request_id = body["data"]["request_id"]

    # Step 2 — TOTP (wait if near 30s boundary)
    totp = pyotp.TOTP(totp_secret)
    if (30 - int(time.time()) % 30) <= 3:
        time.sleep(4)
    r2 = s.post("https://kite.zerodha.com/api/twofa",
                data={"user_id": user_id, "request_id": request_id,
                      "twofa_value": totp.now(), "twofa_type": "totp"}, timeout=15)
    r2.raise_for_status()
    if r2.json().get("status") != "success":
        log.error("TOTP failed."); sys.exit(1)

    # Step 3 — get request_token from redirect
    r3 = s.get(kite.login_url(), allow_redirects=False, timeout=15)
    loc = r3.headers.get("Location", "")
    if "request_token=" not in loc:
        r3 = s.get(loc, allow_redirects=False, timeout=15)
        loc = r3.headers.get("Location", "")
    if "request_token=" not in loc:
        log.error(f"No request_token in redirect: {loc}"); sys.exit(1)
    request_token = loc.split("request_token=")[1].split("&")[0]

    # Step 4 — exchange for access_token
    token_data = kite.generate_session(request_token, api_secret=api_secret)
    kite.set_access_token(token_data["access_token"])
    _save_token(api_key, token_data["access_token"])
    log.info("Auto-login complete.")
    return kite
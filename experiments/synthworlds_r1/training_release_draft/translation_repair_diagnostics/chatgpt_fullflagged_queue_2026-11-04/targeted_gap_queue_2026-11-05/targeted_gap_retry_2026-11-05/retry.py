#!/usr/bin/env python3
"""Retry only the 17 blocked diagnostic rows; never approves release."""
import base64
import gc
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.51.3", "accelerate==1.15.0",
    "lm-format-enforcer==0.11.1", "bitsandbytes==0.50.2",
], check=True)

import torch
import transformers
import accelerate
import bitsandbytes
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

INPUT_B64 = "eyJhbnN3ZXJzX29yX3NwbGl0X2xhYmVsc19pbmNsdWRlZCI6ZmFsc2UsImh1bWFuX3RyYW5zbGF0aW9uX3Jldmlld2VkIjpmYWxzZSwibW9kZSI6IkFJX09OTFlfRElBR05PU1RJQ19SRVRSWV9OT19SRUxFQVNFX0FQUFJPVkFMIiwicHJpb3JfbmxsYl9tb2RlbCI6eyJpZCI6ImZhY2Vib29rL25sbGItMjAwLWRpc3RpbGxlZC02MDBNIiwicmV2aXNpb24iOiJmOGQzMzNhMDk4ZDE5YjRmZDlhOGIxOGY5NDE3MDQ4N2FkM2Y4MjFkIn0sInByaW9yX291dHB1dF9zaGEyNTYiOiI1MjdiNTlkYjQzNzQ3NWMwNTcxNDQxMTEzNjdlZmMxMjJjYTc4MWNjYjk2N2NmNGFlNzQxZThmMmQ3MjkyMTAwIiwicHJpb3JfcXdlbl9tb2RlbCI6eyJpZCI6IlF3ZW4vUXdlbjIuNS0xNEItSW5zdHJ1Y3QiLCJyZXZpc2lvbiI6ImNmOThmM2IzYmJiNDU3YWQ5ZTJiYjdiYWY5YTAxMjViNmI4OGNhYTgifSwicmVsZWFzZV9hcHByb3ZlZCI6ZmFsc2UsInJvd3MiOlt7ImtpbmQiOiJhdG9taWMiLCJsYW5ndWFnZSI6ImRlIiwicmV0cnlfYmFja3RyYW5zbGF0aW9uIjoiQUxJQVNfTkxMQiIsInJldmlld19rZXkiOiJHMDAxOCIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IldoYXQgaXMgdGhlIFtFTlRJVFlfQV0gZGlmZmVyZW50IGZyb20/Iiwic291cmNlX3NoYTI1NiI6IjE2M2FkMDAxMzk2NzhhYjc0MjQyZWYyZTk1ZmExZDkwOTBhZDhhZTU1NmU3YzcxMDJjZjE0N2FmZjQ2ZTYyNmQiLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiV292b24gdW50ZXJzY2hlaWRldCBzaWNoIFtFTlRJVFlfQV0/IiwidHJhbnNsYXRpb25fc2hhMjU2IjoiM2UwNWFkNDQyNzJhZTAyNjNjYjQzN2M4ZWFkYjI5YTdhOWIyNDA2ZGE1ODRhMzFjM2Y1NTg1YjdhMTg2OWQ3MCJ9LHsia2luZCI6ImNvbXBvc2l0aW9uIiwibGFuZ3VhZ2UiOiJoaSIsInJldHJ5X2JhY2t0cmFuc2xhdGlvbiI6IkFMSUFTX05MTEIiLCJyZXZpZXdfa2V5IjoiRzAwMjMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJXaGljaCBwb2xpdGljYWwgb2ZmaWNlIG9mIHRoZSBzdGF0ZSB0aGF0IHJlcGxhY2VkIHRoZSBjb3VudHJ5IHdob3NlIGNhcGl0YWwgaXMgW0VOVElUWV9BXSBpcyBoZWxkIGJ5IGl0cyBoZWFkIG9mIHN0YXRlPyIsInNvdXJjZV9zaGEyNTYiOiI0YTJhOTdjNTlkMWE2YWMxZjliYmU1MGFiNjBhZDgwMmQyM2ZlNDhhYWQ5ZmMwMjg2ZDRiZTIyZDExYmQzNmYxIiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IuCkieCkuCDgpLDgpL7gpJzgpY3gpK8g4KSV4KS+IOCkleCljOCkqC3gpLjgpL4g4KSw4KS+4KSc4KSo4KWA4KSk4KS/4KSVIOCkquCkpiDgpIngpLjgpJXgpYcg4KSw4KS+4KS34KWN4KSf4KWN4KSw4KS+4KSn4KWN4KSv4KSV4KWN4KS3IOCkleClhyDgpKrgpL7gpLgg4KS54KWILCDgpJzgpL/gpLjgpKjgpYcg4KSJ4KS4IOCkpuClh+CktiDgpJXgpL4g4KS44KWN4KSl4KS+4KSoIOCksuCkv+Ckr+CkviDgpJzgpL/gpLjgpJXgpYAg4KSw4KS+4KSc4KSn4KS+4KSo4KWAIFtFTlRJVFlfQV0g4KS54KWIPyIsInRyYW5zbGF0aW9uX3NoYTI1NiI6ImJmYzE5YjZmODFhZTMzOTA4MWRhNTYwZDQzNGExMWNhOTJiMmE3NzBlOTI3MmE3NDA0ODg3NDc4N2YyYzhiMmUifSx7ImtpbmQiOiJhdG9taWMiLCJsYW5ndWFnZSI6InRoIiwicmV0cnlfYmFja3RyYW5zbGF0aW9uIjoiQUxJQVNfTkxMQiIsInJldmlld19rZXkiOiJHMDAyNiIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IkluIHdoYXQgeWVhciBkaWQgW0VOVElUWV9BXSByZWNlaXZlIHRoZSBbRU5USVRZX0JdPyIsInNvdXJjZV9zaGEyNTYiOiJjMjY4ZTY4NmIwNjI0NGVkMDJmOGE5MzI3YjM3ZDA5NTMyOTczNzZlMDk4ODk5M2RmY2VmY2I4MWZkYmNhOTJlIiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IltFTlRJVFlfQV0g4LmE4LiU4LmJ4Lij4Lix4LiaIFtFTlRJVFlfQl0g4LmD4LiZ4Lib4Li14LmD4LiUPyIsInRyYW5zbGF0aW9uX3NoYTI1NiI6Ijk0YzU2ZmRkNTY3N2U2M2FkMmE5MWI1NjE3NDVkYTc4MWJlMjRjNTAwMTgzM2VlYjA5NGMyN2Y1NzQyOWZmYmMifSx7ImtpbmQiOiJhdG9taWMiLCJsYW5ndWFnZSI6ImhpIiwicmV0cnlfYmFja3RyYW5zbGF0aW9uIjoiQUxJQVNfTkxMQiIsInJldmlld19rZXkiOiJHMDA1NSIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IkluIHdoYXQgeWVhciBkaWQgW0VOVElUWV9BXSBzdG9wIHNlcnZpbmcgYXMgYSBib2FyZCBtZW1iZXIgb2YgW0VOVElUWV9CXT8iLCJzb3VyY2Vfc2hhMjU2IjoiZTUzYjY4MjgwNjA4ZmRhOTQyZmRjMDNlMmJkZjgzZDVjMzk1Nzk5N2JkOGEwNzU5YTdlNTg1OGMyZWQzYjk4NCIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiLgpJXgpL/gpLgg4KS14KSw4KWN4KS3IFtFTlRJVFlfQV0g4KSo4KWHIFtFTlRJVFlfQl0g4KSV4KWHIOCkrOCli+CksOCljeCkoSDgpLjgpKbgpLjgpY3gpK8g4KSV4KWHIOCksOClguCkqiDgpK7gpYfgpIIg4KSV4KS+4KSuIOCkleCksOCkqOCkviDgpKzgpILgpKYg4KSV4KS/4KSv4KS+PyIsInRyYW5zbGF0aW9uX3NoYTI1NiI6IjBhZjI1NGYxZDNmMWVlZjU2OTlhZDcxMGNhYmFmZGJkMTEzYWJiYWJhNjZiZjU0YjQxMmEzYmUxNzdkMGNiNWEifSx7ImtpbmQiOiJhdG9taWMiLCJsYW5ndWFnZSI6InRoIiwicmV0cnlfYmFja3RyYW5zbGF0aW9uIjoiQUxJQVNfTkxMQiIsInJldmlld19rZXkiOiJHMDA1OSIsInNvdXJjZV9xdWVzdGlvbl9lbiI6Ik9uIHdoYXQgZGF0ZSBkaWQgW0VOVElUWV9BXSdzIHRlbnVyZSBhcyBkaXJlY3RvciBvZiBbRU5USVRZX0JdIGVuZD8iLCJzb3VyY2Vfc2hhMjU2IjoiNDkxNjU4OTcxOGI2NTljMzQzYTA4ZmIwYWY1ZmY2ZDE5N2Q1ODNiZmVkZmQ2ZGFhYTY2ODcwYWNmNjNhNGQwOSIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiLguKfguLLguKPguLDguIHguLLguKPguJTguLPguKPguIfguJXguLPguYHguKvguJnguYjguIfguJzguLnguYnguK3guLPguJnguKfguKLguIHguLLguKPguILguK3guIcgW0VOVElUWV9BXSDguYPguJkgW0VOVElUWV9CXSDguKrguLTguYnguJnguKrguLjguJTguKXguIfguYPguJnguKfguLHguJnguJfguLXguYjguYPguJQ/IiwidHJhbnNsYXRpb25fc2hhMjU2IjoiMzY2ZGNlNWZhYjQ0NDZjMWQxZjUwYWE2MjllYTlhZWM1YmQ4OGVlMTJiZDg5OGUwYTRhMjZmNTFjZTJmNDhiYiJ9LHsia2luZCI6ImF0b21pYyIsImxhbmd1YWdlIjoiZGUiLCJyZXRyeV9iYWNrdHJhbnNsYXRpb24iOiJBTElBU19OTExCIiwicmV2aWV3X2tleSI6IkcwMDY5Iiwic291cmNlX3F1ZXN0aW9uX2VuIjoiT24gd2hhdCBkYXRlIGRpZCBbRU5USVRZX0FdIGJlY29tZSB0aGUgY291bnRyeSB0aGF0IFtFTlRJVFlfQl0gaXMgaW4/Iiwic291cmNlX3NoYTI1NiI6ImM2OTRkMzY5YTA4NDY3OTEyMDRiMjBhZGUzNWZiNzc3NmIyYjcwNzJiNmU0OWEyMDdiZWY1ZjQxMTYzZmVmMTgiLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiQW4gd2VsY2hlbSBEYXR1bSB3dXJkZSBbRU5USVRZX0FdIHp1IGRlbSBMYW5kLCBpbiBkZW0gW0VOVElUWV9CXSBsaWVndD8iLCJ0cmFuc2xhdGlvbl9zaGEyNTYiOiJjMWFhZWEyNDIxNTg5ZjYwYWMzODZmYjA3YjcwNmNkMjhiMWMxYWU0NTQwZTQ0ZGE3MDU0M2VmYTU5MDk2MTA3In0seyJraW5kIjoiYXRvbWljIiwibGFuZ3VhZ2UiOiJkZSIsInJldHJ5X2JhY2t0cmFuc2xhdGlvbiI6IkFMSUFTX05MTEIiLCJyZXZpZXdfa2V5IjoiRzAwNzIiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJXaGljaCBlbnRpdHkgaXMgW0VOVElUWV9BXSBkaWZmZXJlbnQgZnJvbT8iLCJzb3VyY2Vfc2hhMjU2IjoiYmVmZjYyM2RmNzVjZTZiMWVhNWJkZjQ1MWI2YWYyM2QxZTBiYjEzYjA5NzQ5MTc3YzMxN2U1N2E0NTViZmZjMiIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJWb24gd2VsY2hlciBFbnRpdMOkdCB1bnRlcnNjaGVpZGV0IHNpY2ggW0VOVElUWV9BXT8iLCJ0cmFuc2xhdGlvbl9zaGEyNTYiOiJhNTVmZDc5YjgxM2Y4MWZiNTY0ZmM2ZjUyZjY4MTlkMTE5MzBkODk4NWRjYmFiY2Q5NWZmMDY4ZmVkZTc1YjU2In0seyJraW5kIjoiYXRvbWljIiwibGFuZ3VhZ2UiOiJoaSIsInByaW9yX2JhY2t0cmFuc2xhdGlvbl9lbiI6IldoYXQgY29tZXMgYWZ0ZXIgW0VOVElUWV9BXT8iLCJwcmlvcl9iYWNrdHJhbnNsYXRpb25fc2hhMjU2IjoiYmJlZjFkZDg4ZWNkYjFkZDZkM2NhNTM0Y2MxZWUyZWI5YWVmMWJkNWI0ZThhNDVhM2FmMDNjZmE0YzA4ZjczZCIsInJldHJ5X2JhY2t0cmFuc2xhdGlvbiI6IlJFVVNFX1ZFUklGSUVEX1BSSU9SIiwicmV2aWV3X2tleSI6IkcwMDgzIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiV2hhdCBpbW1lZGlhdGVseSBmb2xsb3dzIFtFTlRJVFlfQV0/Iiwic291cmNlX3NoYTI1NiI6IjFiN2QzMDJhNGNiNzBhYjI3ZDFlOTg5NjlmM2Q3OWU4MTIxNDY5YzQ5MDZkYzYzYWVhNTc1MzY4ZTg5ZTI2NGYiLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiW0VOVElUWV9BXSDgpJXgpYcg4KSk4KWB4KSw4KSC4KSkIOCkrOCkvuCkpiDgpJXgpY3gpK/gpL4g4KSG4KSk4KS+IOCkueCliD8iLCJ0cmFuc2xhdGlvbl9zaGEyNTYiOiI2ZGViYzkyNjg0ZGMxNTQ5YTc0NjM4ZWIwNDI3ODI5ZGQ3ZjdkMTZlNDc5OTgwMDQzYmM2NTY5NjliYTkzOGMwIn0seyJraW5kIjoiYXRvbWljIiwibGFuZ3VhZ2UiOiJoaSIsInJldHJ5X2JhY2t0cmFuc2xhdGlvbiI6IkFMSUFTX05MTEIiLCJyZXZpZXdfa2V5IjoiRzAwODUiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJPbiB3aGF0IGRhdGUgZGlkIFtFTlRJVFlfQV0gYmVjb21lIGEgcmVsYXRpdmUgb2YgW0VOVElUWV9CXT8iLCJzb3VyY2Vfc2hhMjU2IjoiODEyZGFmMTFjNDY2ZWUxMDQyNDc5ODllODAzZmQxYTEzZjEzMGNiZmNlOTA2OGQzNWE5M2E3MDViMWYxM2QwMyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiLgpJXgpL/gpLgg4KSk4KS+4KSw4KWA4KSWIOCkleCliyBbRU5USVRZX0FdIOCklOCksCBbRU5USVRZX0JdIOCksOCkv+CktuCljeCkpOClh+CkpuCkvuCksCDgpKzgpKjgpYc/IiwidHJhbnNsYXRpb25fc2hhMjU2IjoiMmI1ZTI4NDJkOGY0ZTBhYTNiNmQyM2MyODU0YWU5NDdmNWRiNjVlMDVkMDJmOTViYjZlOTEzMTNlOGY2MTgxMSJ9LHsia2luZCI6ImF0b21pYyIsImxhbmd1YWdlIjoiaGkiLCJyZXRyeV9iYWNrdHJhbnNsYXRpb24iOiJBTElBU19OTExCIiwicmV2aWV3X2tleSI6IkcwMDkwIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiT24gd2hhdCBkYXRlIGRpZCBbRU5USVRZX0FdIGJlY29tZSB0aGUgY291bnRyeSB0aGF0IFtFTlRJVFlfQl0gaXMgaW4/Iiwic291cmNlX3NoYTI1NiI6ImM2OTRkMzY5YTA4NDY3OTEyMDRiMjBhZGUzNWZiNzc3NmIyYjcwNzJiNmU0OWEyMDdiZWY1ZjQxMTYzZmVmMTgiLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoi4KSV4KS/4KS4IOCkpOCkvuCksOClgOCkliDgpJXgpYsgW0VOVElUWV9BXSDgpLXgpLkg4KSm4KWH4KS2IOCkrOCkqOCkviDgpJzgpL/gpLjgpK7gpYfgpIIgW0VOVElUWV9CXSDgpLjgpY3gpKXgpL/gpKQg4KS54KWIPyIsInRyYW5zbGF0aW9uX3NoYTI1NiI6IjFkYzMwNGZiZDFkYTE4MTI2YTY0MzUxNGJjMzE5NjdmZjkxZWIwZWFkZDQ5OGZjZjZhNzAwYTczNmU5ZjZmYzYifSx7ImtpbmQiOiJhdG9taWMiLCJsYW5ndWFnZSI6ImhpIiwicmV0cnlfYmFja3RyYW5zbGF0aW9uIjoiQUxJQVNfTkxMQiIsInJldmlld19rZXkiOiJHMDA5MiIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IkluIHdoYXQgeWVhciBkaWQgW0VOVElUWV9BXSBiZWNvbWUgdGhlIHNwb3VzZSBvZiBbRU5USVRZX0JdPyIsInNvdXJjZV9zaGEyNTYiOiJiY2UyYjJlYzYyYTkxMDhmOWU5ZmY3ZjUyZDhlODFhMTEzMjBkNjg1ODA0MDkwNjgyYjViNGIzMDRlNDE5ODI0IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IuCkleCkv+CkuCDgpLXgpLDgpY3gpLcgW0VOVElUWV9BXSDgpJTgpLAgW0VOVElUWV9CXSDgpJzgpYDgpLXgpKjgpLjgpL7gpKXgpYAg4KSs4KSo4KWHPyIsInRyYW5zbGF0aW9uX3NoYTI1NiI6IjdhNmRmYzU3ZjAyNGUwY2VlNjhjZjlkNGJmOWFkNThhM2UwOTU2MzQwOWJjMTUwZjdhNjBkMTg3MDEyYjUxMTgifSx7ImtpbmQiOiJjb21wb3NpdGlvbiIsImxhbmd1YWdlIjoiaGkiLCJyZXRyeV9iYWNrdHJhbnNsYXRpb24iOiJBTElBU19OTExCIiwicmV2aWV3X2tleSI6IkcwMDk2Iiwic291cmNlX3F1ZXN0aW9uX2VuIjoiV2hhdCBwYXltZW50IHR5cGVzIGFyZSBhY2NlcHRlZCBieSB0aGUgb3JnYW5pemVyIG9mIHRoZSBldmVudCBbRU5USVRZX0FdIHBhcnRpY2lwYXRlZCBpbj8iLCJzb3VyY2Vfc2hhMjU2IjoiMGMwOGUwZGNkMTdhNzc3MTM4OGY5NTliNDk1YTkzNzc4YzViMWIzZDMwYjY0MzNlMTU2NjFjNWYwNjQyN2MyYyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiLgpJzgpL/gpLgg4KSG4KSv4KWL4KSc4KSoIOCkruClh+CkgiBbRU5USVRZX0FdIOCkqOClhyDgpK3gpL7gpJcg4KSy4KS/4KSv4KS+LCDgpIngpLjgpJXgpYcg4KSG4KSv4KWL4KSc4KSVIOCkleCljOCkqC3gpLjgpYcg4KSt4KWB4KSX4KSk4KS+4KSoIOCkquCljeCksOCkleCkvuCksCDgpLjgpY3gpLXgpYDgpJXgpL7gpLAg4KSV4KSw4KSk4KWHIOCkueCliOCkgj8iLCJ0cmFuc2xhdGlvbl9zaGEyNTYiOiI0YTlhN2MyNjkwM2U3NmNkZTgyMjc0ODI5NTk4NGI5NGY3YTU5MTRmZDk5OGU5Mzc1ZWYyMWE1YTc5ZTY2ZDRmIn0seyJraW5kIjoiYXRvbWljIiwibGFuZ3VhZ2UiOiJoaSIsInJldHJ5X2JhY2t0cmFuc2xhdGlvbiI6IkFMSUFTX05MTEIiLCJyZXZpZXdfa2V5IjoiRzAxMTMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJPbiB3aGF0IGRhdGUgZGlkIFtFTlRJVFlfQV0gY2Vhc2UgdG8gYmUgdGhlIGNvdW50cnkgb2YgW0VOVElUWV9CXT8iLCJzb3VyY2Vfc2hhMjU2IjoiZGI5YmQ2NjkwYWE0MDQyZmNiMmNhZDRmZGY3OTg3ZDk3YzczMmQ0N2FhNzgyNjVhOGI2NDE1ZDFjYjE2MzE1MiIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiLgpJXgpL/gpLgg4KSk4KS+4KSw4KWA4KSWIOCkleCliyBbRU5USVRZX0FdLCBbRU5USVRZX0JdIOCkleCkviDgpKbgpYfgpLYg4KS54KWL4KSo4KS+IOCkrOCkguCkpiDgpLngpYHgpIY/IiwidHJhbnNsYXRpb25fc2hhMjU2IjoiN2U1MGUyNWMzOWFmNmIwMGRkNTI2NWUxZTE4MzAwODMxZjU1MmJlYzFmMDg5OTllMDY3YzY2NTY4NDZlNTc3YiJ9LHsia2luZCI6ImF0b21pYyIsImxhbmd1YWdlIjoiaGkiLCJyZXRyeV9iYWNrdHJhbnNsYXRpb24iOiJBTElBU19OTExCIiwicmV2aWV3X2tleSI6IkcwMTQyIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiT24gd2hhdCBkYXRlIGRpZCBbRU5USVRZX0FdIHN0b3AgYmVpbmcgdGhlIGNhcGl0YWwgb2YgW0VOVElUWV9CXT8iLCJzb3VyY2Vfc2hhMjU2IjoiZjAwZGQzOTZiM2FhN2ViNDVlMGZhOGM4MjhmOGY2OTMzMjkwNDc1N2JlYjUxMDVmMDFjNTA2MGI4NWFjNDJiOSIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiLgpJXgpL/gpLgg4KSk4KS+4KSw4KWA4KSWIOCkleCliyBbRU5USVRZX0FdIOCkqOClhyBbRU5USVRZX0JdIOCkleClgCDgpLDgpL7gpJzgpKfgpL7gpKjgpYAg4KS54KWL4KSo4KS+IOCkrOCkguCkpiDgpJXgpL/gpK/gpL4/IiwidHJhbnNsYXRpb25fc2hhMjU2IjoiZjg4YjYyOGIyZmU1MDhlNDIxOTIxMjQyNTJjNDk0ZGZmZTg2ZjMyM2QwMDY4MzlhMzdlYTIxMzE3NzYyMDIyYiJ9LHsia2luZCI6ImF0b21pYyIsImxhbmd1YWdlIjoiZnIiLCJyZXRyeV9iYWNrdHJhbnNsYXRpb24iOiJBTElBU19OTExCIiwicmV2aWV3X2tleSI6IkcwMTQ0Iiwic291cmNlX3F1ZXN0aW9uX2VuIjoiT24gd2hhdCBkYXRlIGRpZCBbRU5USVRZX0FdIGJlY29tZSBhIHJlbGF0aXZlIG9mIFtFTlRJVFlfQl0/Iiwic291cmNlX3NoYTI1NiI6IjgxMmRhZjExYzQ2NmVlMTA0MjQ3OTg5ZTgwM2ZkMWExM2YxMzBjYmZjZTkwNjhkMzVhOTNhNzA1YjFmMTNkMDMiLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiw4AgcXVlbGxlIGRhdGUgbGUgbGllbiBkZSBwYXJlbnTDqSBlbnRyZSBbRU5USVRZX0FdIGV0IFtFTlRJVFlfQl0gYS10LWlsIGNvbW1lbmPDqSA/IiwidHJhbnNsYXRpb25fc2hhMjU2IjoiNWVhMGNmODczZjBhNDE4NDI2ZDQ5MTYwZGRiZDRlYjExZTZlZDUwMzk5ODg1ZWYzMTk0NWIwMTc3Nzk2ZjkxZiJ9LHsia2luZCI6ImNvbXBvc2l0aW9uIiwibGFuZ3VhZ2UiOiJoaSIsInByaW9yX2JhY2t0cmFuc2xhdGlvbl9lbiI6IldoYXQgaXMgdGhlIHRpdGxlIG9mIHRoZSBbRU5USVRZX0FdPyIsInByaW9yX2JhY2t0cmFuc2xhdGlvbl9zaGEyNTYiOiJmYzQ3NGVjMzIwNTViMjQ0ODRmN2I1YjdhNTFmNWUyOTA2YmNlOGFmMjIyZTc3N2Y5OTkzOWVlYjQ2M2NkMjNkIiwicmV0cnlfYmFja3RyYW5zbGF0aW9uIjoiUkVVU0VfVkVSSUZJRURfUFJJT1IiLCJyZXZpZXdfa2V5IjoiRzAxNjIiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJXaGljaCBub2JsZSB0aXRsZSBkb2VzIFtFTlRJVFlfQV0gaG9sZD8iLCJzb3VyY2Vfc2hhMjU2IjoiMjg5OGYwNGY2Yzc0YmE3NGJiMDM2MDJkNTQ0ZTBjNjM5MTMxZTQ1ZmI3YjI4MWNiY2ZjNGNkZDcwZGQ3MzFjZiIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJbRU5USVRZX0FdIOCkleClhyDgpKrgpL7gpLgg4KSV4KWM4KSoLeCkuOClgCDgpJXgpYHgpLLgpYDgpKgg4KSJ4KSq4KS+4KSn4KS/IOCkueCliD8iLCJ0cmFuc2xhdGlvbl9zaGEyNTYiOiJmMDM1M2I2NmU3YzgyMzY2ZWVlYzc5MThiYjViZGE4ZDBkNjM1MTgwYTU0NjM1ZWM2NjA4NTkyMDk1YTBmN2NkIn0seyJraW5kIjoiYXRvbWljIiwibGFuZ3VhZ2UiOiJmciIsInJldHJ5X2JhY2t0cmFuc2xhdGlvbiI6IkFMSUFTX05MTEIiLCJyZXZpZXdfa2V5IjoiRzAxNzYiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJJbiB3aGF0IHllYXIgd2FzIFtFTlRJVFlfQV0gdGhlIGFyY2hpdGVjdCBvZiBbRU5USVRZX0JdPyIsInNvdXJjZV9zaGEyNTYiOiI4ZGNlYTliY2IzNzRkZjU4NTFlMmM0NWRmYTliNGU0YzFhNTdmYzQzM2YyNjYyYzBhMDkxYWFlYzQ2MDZiNDNlIiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IkVuIHF1ZWxsZSBhbm7DqWUgW0VOVElUWV9BXSBhLXQtaWwgw6l0w6kgbOKAmWFyY2hpdGVjdGUgZGUgW0VOVElUWV9CXSA/IiwidHJhbnNsYXRpb25fc2hhMjU2IjoiM2U3ODg1ZjJmOWJhMjk4ODFiMmEwZDUwN2Y5ZTRlNjQyMzU0ZGJmZGJlZjc1OWMwOWZlYzA2YmUwNTg0M2E2OCJ9XSwic2NoZW1hIjoic3ludGh3b3JsZHMtdGFyZ2V0ZWQtZ2FwLXJldHJ5LXYxIiwic291cmNlX2lucHV0X3NoYTI1NiI6IjJmZGE3NjViY2Q4NzMzMTgxZTc1ZDdjMWQzZjFkN2FhNjlhYWRiN2Q5ZTg3YjMxZmFjYzk1ZTVkYWVhMTNhYmYiLCJ0cmFpbmluZ19lbGlnaWJsZSI6ZmFsc2V9Cg=="
INPUT_SHA256 = "11d529966b7042958687c0d7ef69a2de0e4e250775e4d31bfd65379cf2493f7b"
NLLB_ID = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
QWEN_ID = "Qwen/Qwen2.5-14B-Instruct"
QWEN_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
OUT = Path("/kaggle/working/retry_output.json")
LANGUAGE_NAMES = {"fr": "French", "de": "German", "hi": "Hindi", "th": "Thai"}
NLLB_SOURCE_CODES = {"fr": "fra_Latn", "de": "deu_Latn", "hi": "hin_Deva", "th": "tha_Thai"}
MARKER = re.compile(r"\[(ENTITY|NUMBER)_([A-Z]+)\]")
ALIAS = {"[ENTITY_A]": "Alice", "[ENTITY_B]": "Bob"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_input():
    raw = base64.b64decode(INPUT_B64, validate=True)
    if sha(raw) != INPUT_SHA256:
        raise ValueError("retry input SHA-256 mismatch")
    payload = json.loads(raw, object_pairs_hook=unique_object)
    if (payload.get("schema") != "synthworlds-targeted-gap-retry-v1"
            or payload.get("mode") != "AI_ONLY_DIAGNOSTIC_RETRY_NO_RELEASE_APPROVAL"
            or payload.get("human_translation_reviewed") is not False
            or payload.get("release_approved") is not False
            or payload.get("training_eligible") is not False
            or payload.get("answers_or_split_labels_included") is not False):
        raise ValueError("retry input scope/approval flags are invalid")
    rows = payload.get("rows")
    if (not isinstance(rows, list) or len(rows) != 17
            or len({row.get("review_key") for row in rows}) != 17):
        raise ValueError("expected exactly 17 unique blinded retry rows")
    for row in rows:
        if row["language"] not in LANGUAGE_NAMES or row["kind"] not in {"atomic", "composition"}:
            raise ValueError(f"invalid language/kind for {row.get('review_key')}")
        if sha(row["source_question_en"].encode()) != row["source_sha256"]:
            raise ValueError(f"source hash mismatch: {row['review_key']}")
        if sha(row["translated_question"].encode()) != row["translation_sha256"]:
            raise ValueError(f"candidate hash mismatch: {row['review_key']}")
        if row["retry_backtranslation"] not in {"ALIAS_NLLB", "REUSE_VERIFIED_PRIOR"}:
            raise ValueError(f"invalid retry mode: {row['review_key']}")
    return payload, rows


def mask_aliases(text):
    originals, expected = {}, Counter()
    def replace(match):
        marker = match.group(0)
        alias = ALIAS.get(marker)
        if alias is None:
            raise ValueError(f"no approved alias for placeholder {marker}")
        if alias.casefold() in text.casefold().replace(marker.casefold(), ""):
            raise ValueError(f"alias collision for placeholder {marker}")
        originals[alias] = marker
        expected[alias] += 1
        return alias
    masked = MARKER.sub(replace, text)
    if not expected:
        raise ValueError("candidate contains no protected placeholder")
    return masked, originals, expected


def restore_aliases(text, originals, expected):
    observed = Counter({alias: text.count(alias) for alias in originals})
    if observed != expected:
        raise ValueError(f"alias multiplicities changed: expected={dict(expected)}, observed={dict(observed)}")
    for alias, marker in originals.items():
        text = text.replace(alias, marker)
    if any(alias in text for alias in originals):
        raise ValueError("one or more aliases were not restored")
    return text.strip()


def schema():
    decision = {"type": "object", "properties": {
        "key": {"type": "string"}, "semantic_pass": {"type": "boolean"},
        "fluency_pass": {"type": "boolean"}, "reason": {"type": "string", "maxLength": 120}},
        "required": ["key", "semantic_pass", "fluency_pass", "reason"],
        "additionalProperties": False}
    return {"type": "object", "properties": {
        "decisions": {"type": "array", "items": decision, "minItems": 1, "maxItems": 1}},
        "required": ["decisions"], "additionalProperties": False}


def judge(row, backtranslation, model, tokenizer, prefix_function, max_new_tokens):
    system = (
        "Adjudicate this synthetic QA question translation. Do not answer it or infer facts. "
        "Compare the English source, target text, and English back-translation. Accept faithful paraphrases. "
        "semantic_pass is false only for a material change to relation, direction, polarity, scope, time, "
        "or answer type. fluency_pass judges grammar and naturalness in the named target language. "
        "If uncertain about a material defect, return false. Give a brief reason of at most 12 words. "
        "Return one decision for the exact supplied key."
    )
    payload = {"target_language": LANGUAGE_NAMES[row["language"]], "items": [{
        "key": row["review_key"], "source_question_en": row["source_question_en"],
        "translated_question": row["translated_question"], "back_translation_en": backtranslation}]}
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id,
                                   prefix_allowed_tokens_fn=prefix_function)
    raw = tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    start = raw.find("{")
    if start < 0:
        raise ValueError("judge output contains no JSON object")
    parsed, end = json.JSONDecoder(object_pairs_hook=unique_object).raw_decode(raw, start)
    if raw[end:].strip():
        raise ValueError("judge output contains trailing non-JSON text")
    decisions = parsed.get("decisions") if isinstance(parsed, dict) else None
    if not isinstance(decisions, list) or len(decisions) != 1:
        raise ValueError("judge output has invalid decision coverage")
    decision = decisions[0]
    if (not isinstance(decision, dict) or decision.get("key") != row["review_key"]
            or type(decision.get("semantic_pass")) is not bool
            or type(decision.get("fluency_pass")) is not bool
            or not isinstance(decision.get("reason"), str) or not decision["reason"].strip()
            or len(decision["reason"]) > 120):
        raise ValueError("judge output violates the bounded decision schema or key")
    return raw, decision


def atomic_save(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main():
    payload, rows = load_input()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; refusing a CPU fallback")
    runtime = {"status": "RUNNING_AI_ONLY_DIAGNOSTIC_RETRY", "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "torch": torch.__version__, "transformers": transformers.__version__,
               "accelerate": accelerate.__version__, "bitsandbytes": bitsandbytes.__version__,
               "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
               "quantization": "NF4, double quantization, float16 compute",
               "human_translation_reviewed": False, "release_approved": False,
               "training_eligible": False, "answers_or_split_labels_included": False}
    records = []
    needs_nllb = any(r["retry_backtranslation"] == "ALIAS_NLLB" for r in rows)
    bt_tokenizer = bt_model = None
    if needs_nllb:
        bt_tokenizer = NllbTokenizerFast.from_pretrained(NLLB_ID, revision=NLLB_REVISION)
        bt_model = AutoModelForSeq2SeqLM.from_pretrained(
            NLLB_ID, revision=NLLB_REVISION, torch_dtype=torch.float16, device_map="auto")
        bt_model.eval()
    record_by_key = {}
    for index, row in enumerate(rows, 1):
        record = {"review_key": row["review_key"], "language": row["language"], "kind": row["kind"],
                  "source_sha256": row["source_sha256"], "translation_sha256": row["translation_sha256"],
                  "retry_backtranslation": row["retry_backtranslation"], "backtranslation_en": "",
                  "backtranslation_sha256": None, "backtranslation_error": None,
                  "decision": None, "judge_attempts": [], "raw_output": None}
        try:
            if row["retry_backtranslation"] == "REUSE_VERIFIED_PRIOR":
                text = row["prior_backtranslation_en"]
                if sha(text.encode()) != row["prior_backtranslation_sha256"]:
                    raise ValueError("prior backtranslation hash mismatch")
                record["backtranslation_en"] = text
                record["backtranslation_sha256"] = sha(text.encode())
            else:
                source_markers = Counter(MARKER.findall(row["source_question_en"]))
                target_markers = Counter(MARKER.findall(row["translated_question"]))
                if source_markers != target_markers:
                    raise ValueError("source/candidate placeholder multiplicities differ")
                protected, originals, expected = mask_aliases(row["translated_question"])
                bt_tokenizer.src_lang = NLLB_SOURCE_CODES[row["language"]]
                encoded = bt_tokenizer(protected, return_tensors="pt", truncation=True, max_length=256).to(bt_model.device)
                with torch.inference_mode():
                    generated = bt_model.generate(**encoded,
                        forced_bos_token_id=bt_tokenizer.convert_tokens_to_ids("eng_Latn"),
                        max_new_tokens=128, do_sample=False)
                translated = bt_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
                restored = restore_aliases(translated, originals, expected)
                if Counter(MARKER.findall(restored)) != source_markers:
                    raise ValueError("restored marker multiset differs from source")
                record["backtranslation_en"] = restored
                record["backtranslation_sha256"] = sha(restored.encode())
        except Exception as exc:
            record["backtranslation_error"] = f"{type(exc).__name__}: {exc}"
        record_by_key[row["review_key"]] = record
        records.append(record)
        atomic_save(OUT, {"status": "IN_PROGRESS_ALIAS_BACKTRANSLATION", "input_sha256": INPUT_SHA256,
            "input_payload": payload, "records": records, **runtime})
        print(f"Backtranslation {index}/{len(rows)}", flush=True)
    if bt_model is not None:
        del bt_model, bt_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, revision=QWEN_REVISION)
    model = AutoModelForCausalLM.from_pretrained(QWEN_ID, revision=QWEN_REVISION,
        torch_dtype=torch.float16, quantization_config=quantization, device_map="auto")
    model.eval()
    prefix_function = build_transformers_prefix_allowed_tokens_fn(tokenizer, JsonSchemaParser(schema()))
    for index, row in enumerate(rows, 1):
        record = record_by_key[row["review_key"]]
        if record["backtranslation_error"]:
            record["decision"] = {"error": "backtranslation_failed; judgment withheld"}
        else:
            for max_new_tokens in (128, 192, 256):
                try:
                    raw, decision = judge(row, record["backtranslation_en"], model, tokenizer,
                                          prefix_function, max_new_tokens)
                    record["raw_output"] = raw
                    record["decision"] = decision
                    record["judge_attempts"].append({"max_new_tokens": max_new_tokens, "status": "VALID"})
                    break
                except Exception as exc:
                    record["judge_attempts"].append({"max_new_tokens": max_new_tokens,
                        "status": "INVALID", "error": f"{type(exc).__name__}: {exc}"})
            if record["decision"] is None:
                record["decision"] = {"error": "judge_invalid_after_three_bounded_retries"}
        atomic_save(OUT, {"status": "IN_PROGRESS_QWEN_RETRY", "input_sha256": INPUT_SHA256,
            "input_payload": payload, "records": records, **runtime})
        print(f"Judge {index}/{len(rows)}", flush=True)

    runtime["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    runtime["reviewed_rows"] = sum("error" not in (r.get("decision") or {}) for r in records)
    runtime["blocked_rows"] = len(records) - runtime["reviewed_rows"]
    runtime["status"] = "AI_ONLY_DIAGNOSTIC_RETRY_COMPLETE" if runtime["blocked_rows"] == 0 else "AI_ONLY_DIAGNOSTIC_RETRY_HAS_BLOCKED_ROWS"
    result = {"status": runtime["status"], "input_sha256": INPUT_SHA256,
        "retry_input_payload": payload, "nllb_model_id": NLLB_ID, "nllb_model_revision": NLLB_REVISION,
        "qwen_model_id": QWEN_ID, "qwen_model_revision": QWEN_REVISION,
        "records": records, **runtime}
    atomic_save(OUT, result)
    print(json.dumps({"status": runtime["status"], "rows": len(rows),
        "reviewed_rows": runtime["reviewed_rows"], "blocked_rows": runtime["blocked_rows"],
        "gpu": runtime["gpu"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

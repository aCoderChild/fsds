#!/usr/bin/env python3
"""Bounded NLLB marker-alias diagnostic. Never approves translation quality."""
import base64, collections, hashlib, json, re
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast

MODEL_ID = "facebook/nllb-200-distilled-600M"
MODEL_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
NLLB_CODES = {"fr": "fra_Latn", "de": "deu_Latn", "hi": "hin_Deva", "th": "tha_Thai"}
ALIASES = {"[ENTITY_A]": "Alice", "[ENTITY_B]": "Bob"}
RAW_INPUT_B64 = "W3sicmV2aWV3X2tleSI6IlIwMDExIiwibGFuZ3VhZ2UiOiJmciIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJPbiB3aGF0IGRhdGUgZGlkIFtFTlRJVFlfQV0gYmVjb21lIGEgcmVsYXRpdmUgb2YgW0VOVElUWV9CXT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiw4AgcXVlbGxlIGRhdGUgW0VOVElUWV9BXSBlc3QtaWwgZGV2ZW51IHVuIG1lbWJyZSBkZSBsYSBmYW1pbGxlIGRlIFtFTlRJVFlfQl0gPyJ9LHsicmV2aWV3X2tleSI6IlIwMDM2IiwibGFuZ3VhZ2UiOiJmciIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJJbiB3aGF0IHllYXIgd2FzIFtFTlRJVFlfQV0gdGhlIGFyY2hpdGVjdCBvZiBbRU5USVRZX0JdPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJFbiBxdWVsbGUgYW5uw6llIFtFTlRJVFlfQV0gw6l0YWl0LWlsIGzigJlhcmNoaXRlY3RlIGRlIFtFTlRJVFlfQl0gPyJ9LHsicmV2aWV3X2tleSI6IlIwMTc5IiwibGFuZ3VhZ2UiOiJkZSIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJPbiB3aGF0IGRhdGUgZGlkIFtFTlRJVFlfQV0gYmVjb21lIHRoZSBjb3VudHJ5IHRoYXQgW0VOVElUWV9CXSBpcyBpbj8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiQW4gd2VsY2hlbSBEYXR1bSB3dXJkZSBbRU5USVRZX0FdIHp1IGRlbSBMYW5kLCBpbiBkZW0gc2ljaCBbRU5USVRZX0JdIGJlZmluZGV0PyJ9LHsicmV2aWV3X2tleSI6IlIwMjA2IiwibGFuZ3VhZ2UiOiJkZSIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJXaGF0IGlzIHRoZSBbRU5USVRZX0FdIGRpZmZlcmVudCBmcm9tPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJXb3ZvbiB1bnRlcnNjaGVpZGV0IHNpY2ggW0VOVElUWV9BXT8ifSx7InJldmlld19rZXkiOiJSMDIyMiIsImxhbmd1YWdlIjoiZGUiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiV2hpY2ggZW50aXR5IGlzIFtFTlRJVFlfQV0gZGlmZmVyZW50IGZyb20/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IlZvbiB3ZWxjaGVyIEVudGl0w6R0IHVudGVyc2NoZWlkZXQgc2ljaCBbRU5USVRZX0FdPyJ9LHsicmV2aWV3X2tleSI6IlIwMjM4IiwibGFuZ3VhZ2UiOiJkZSIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJPbiB3aGF0IGRhdGUgZGlkIFtFTlRJVFlfQV0gc3RvcCBiZWluZyB0aGUgY2FwaXRhbCBvZiBbRU5USVRZX0JdPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJBbiB3ZWxjaGVtIERhdHVtIGjDtnJ0ZSBbRU5USVRZX0FdIGF1ZiwgZGllIEhhdXB0c3RhZHQgdm9uIFtFTlRJVFlfQl0genUgc2Vpbj8ifSx7InJldmlld19rZXkiOiJSMDM0NCIsImxhbmd1YWdlIjoiaGkiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiSW4gd2hhdCB5ZWFyIGRpZCBbRU5USVRZX0FdIGZvdW5kIFtFTlRJVFlfQl0/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IltFTlRJVFlfQV0g4KSo4KWHIFtFTlRJVFlfQl0g4KSV4KWAIOCkuOCljeCkpeCkvuCkquCkqOCkviDgpJXgpL/gpLgg4KS14KSw4KWN4KS3IOCkleClgCDgpKXgpYA/In0seyJyZXZpZXdfa2V5IjoiUjAzNDkiLCJsYW5ndWFnZSI6ImhpIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6Ik9uIHdoYXQgZGF0ZSBkaWQgW0VOVElUWV9BXSBiZWNvbWUgYSByZWxhdGl2ZSBvZiBbRU5USVRZX0JdPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJbRU5USVRZX0FdIOCkleCkv+CkuCDgpKTgpL7gpLDgpYDgpJYg4KSV4KWLIFtFTlRJVFlfQl0g4KSV4KS+IOCksOCkv+CktuCljeCkpOClh+CkpuCkvuCksCDgpKzgpKjgpL4/In0seyJyZXZpZXdfa2V5IjoiUjAzNjAiLCJsYW5ndWFnZSI6ImhpIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6Ik9uIHdoYXQgZGF0ZSBkaWQgW0VOVElUWV9BXSBiZWNvbWUgdGhlIGNvdW50cnkgdGhhdCBbRU5USVRZX0JdIGlzIGluPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiLgpJXgpL/gpLgg4KSk4KS+4KSw4KWA4KSW4KS8IOCkleCliyBbRU5USVRZX0FdIOCkteCkuSDgpKbgpYfgpLYg4KSs4KSo4KS+IOCknOCkv+CkuOCkruClh+CkgiBbRU5USVRZX0JdIOCkuOCljeCkpeCkv+CkpCDgpLngpYg/In0seyJyZXZpZXdfa2V5IjoiUjAzNzEiLCJsYW5ndWFnZSI6ImhpIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IkluIHdoYXQgeWVhciB3YXMgW0VOVElUWV9BXSBub21pbmF0ZWQgZm9yIFtFTlRJVFlfQl0/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IltFTlRJVFlfQV0g4KSV4KWLIFtFTlRJVFlfQl0g4KSV4KWHIOCksuCkv+CkjyDgpJXgpL/gpLgg4KS14KSw4KWN4KS3IOCkqOCkvuCkruCkvuCkguCkleCkv+CkpCDgpJXgpL/gpK/gpL4g4KSX4KSv4KS+IOCkpeCkvj8ifSx7InJldmlld19rZXkiOiJSMDM3OSIsImxhbmd1YWdlIjoiaGkiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiT24gd2hhdCBkYXRlIGRpZCBbRU5USVRZX0FdIGNlYXNlIHRvIGJlIHRoZSBjb3VudHJ5IG9mIFtFTlRJVFlfQl0/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IuCkleCkv+CkuCDgpKTgpL7gpLDgpYDgpJbgpLwg4KSV4KWLIFtFTlRJVFlfQV0sIFtFTlRJVFlfQl0g4KSV4KS+IOCkpuClh+CktiDgpKjgpLngpYDgpIIg4KSw4KS54KS+PyJ9LHsicmV2aWV3X2tleSI6IlIwNDA3IiwibGFuZ3VhZ2UiOiJoaSIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJJbiB3aGF0IHllYXIgZGlkIFtFTlRJVFlfQV0gYmVjb21lIHRoZSBzcG91c2Ugb2YgW0VOVElUWV9CXT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiW0VOVElUWV9BXSDgpJXgpL/gpLgg4KS14KSw4KWN4KS3IFtFTlRJVFlfQl0g4KSV4KS+IOCknOClgOCkteCkqOCkuOCkvuCkpeClgCDgpKzgpKjgpL4/In0seyJyZXZpZXdfa2V5IjoiUjA0MTciLCJsYW5ndWFnZSI6ImhpIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6Ik9uIHdoYXQgZGF0ZSBkaWQgW0VOVElUWV9BXSBzdG9wIGJlaW5nIHRoZSBjYXBpdGFsIG9mIFtFTlRJVFlfQl0/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IuCkleCkv+CkuCDgpKTgpL7gpLDgpYDgpJbgpLwg4KSV4KWLIFtFTlRJVFlfQV0sIFtFTlRJVFlfQl0g4KSV4KWAIOCksOCkvuCknOCkp+CkvuCkqOClgCDgpKjgpLngpYDgpIIg4KSw4KS54KS+PyJ9LHsicmV2aWV3X2tleSI6IlIwNDE4IiwibGFuZ3VhZ2UiOiJoaSIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJJbiB3aGF0IHllYXIgZGlkIFtFTlRJVFlfQV0ncyB0ZXJtIGFzIGhlYWQgb2YgZ292ZXJubWVudCBvZiBbRU5USVRZX0JdIGVuZD8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiW0VOVElUWV9CXSDgpJXgpYcg4KS24KS+4KS44KSo4KS+4KSn4KWN4KSv4KSV4KWN4KS3IOCkleClhyDgpLDgpYLgpKog4KSu4KWH4KSCIFtFTlRJVFlfQV0g4KSV4KS+IOCkleCkvuCksOCljeCkr+CkleCkvuCksiDgpJXgpL/gpLgg4KS14KSw4KWN4KS3IOCkuOCkruCkvuCkquCljeCkpCDgpLngpYHgpIY/In0seyJyZXZpZXdfa2V5IjoiUjA0MzYiLCJsYW5ndWFnZSI6ImhpIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IldoZW4gZGlkIFtFTlRJVFlfQV0gYmVjb21lIHBhcnQgb2YgdGhlIFtFTlRJVFlfQl0/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IltFTlRJVFlfQV0g4KSV4KSsIFtFTlRJVFlfQl0g4KSV4KS+IOCkueCkv+CkuOCljeCkuOCkviDgpKzgpKjgpL4/In0seyJyZXZpZXdfa2V5IjoiUjA0NDIiLCJsYW5ndWFnZSI6ImhpIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IkluIHdoYXQgeWVhciBkaWQgW0VOVElUWV9BXSByZWNlaXZlIHRoZSBbRU5USVRZX0JdPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJbRU5USVRZX0FdIOCkleCliyBbRU5USVRZX0JdIOCkleCkv+CkuCDgpLXgpLDgpY3gpLcg4KSq4KWN4KSw4KS+4KSq4KWN4KSkIOCkueClgeCkhiDgpKXgpL4/In0seyJyZXZpZXdfa2V5IjoiUjA0NDkiLCJsYW5ndWFnZSI6ImhpIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IkluIHdoYXQgeWVhciBkaWQgW0VOVElUWV9BXSBzdG9wIHNlcnZpbmcgYXMgYSBib2FyZCBtZW1iZXIgb2YgW0VOVElUWV9CXT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiW0VOVElUWV9BXSDgpKjgpYcg4KSV4KS/4KS4IOCkteCksOCljeCktyBbRU5USVRZX0JdIOCkleClhyDgpKzgpYvgpLDgpY3gpKEg4KS44KSm4KS44KWN4KSvIOCkleClhyDgpLDgpYLgpKog4KSu4KWH4KSCIOCkleCkvuCksOCljeCkryDgpJXgpLDgpKjgpL4g4KSs4KSC4KSmIOCkleCkv+Ckr+Ckvj8ifSx7InJldmlld19rZXkiOiJSMDQ1NSIsImxhbmd1YWdlIjoiaGkiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiSW4gd2hhdCB5ZWFyIHdhcyBbRU5USVRZX0FdIGEgZG9jdG9yYWwgc3R1ZGVudCBvZiBbRU5USVRZX0JdPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJbRU5USVRZX0FdIOCkleCkv+CkuCDgpLXgpLDgpY3gpLcgW0VOVElUWV9CXSDgpJXgpL4g4KSh4KWJ4KSV4KWN4KSf4KSw4KWH4KSfIOCkm+CkvuCkpOCljeCksCDgpKXgpL4/In0seyJyZXZpZXdfa2V5IjoiUjA0ODEiLCJsYW5ndWFnZSI6ImhpIiwia2luZCI6ImNvbXBvc2l0aW9uIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiV2hpY2ggcG9saXRpY2FsIG9mZmljZSBvZiB0aGUgc3RhdGUgdGhhdCByZXBsYWNlZCB0aGUgY291bnRyeSB3aG9zZSBjYXBpdGFsIGlzIFtFTlRJVFlfQV0gaXMgaGVsZCBieSBpdHMgaGVhZCBvZiBzdGF0ZT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoi4KSc4KS/4KS4IOCkpuClh+CktiDgpJXgpYAg4KSw4KS+4KSc4KSn4KS+4KSo4KWAIFtFTlRJVFlfQV0g4KS54KWILCDgpIngpLjgpJXgpL4g4KS44KWN4KSl4KS+4KSoIOCksuClh+CkqOClhyDgpLXgpL7gpLLgpYcg4KSw4KS+4KSc4KWN4KSvIOCkleClhyDgpLDgpL7gpLfgpY3gpJ/gpY3gpLDgpL7gpKfgpY3gpK/gpJXgpY3gpLcg4KSV4KWHIOCkquCkvuCkuCDgpIngpLgg4KSw4KS+4KSc4KWN4KSvIOCkleCkviDgpJXgpYzgpKgt4KS44KS+IOCksOCkvuCknOCkqOClgOCkpOCkv+CklSDgpKrgpKYg4KS54KWIPyJ9LHsicmV2aWV3X2tleSI6IlIwNDg5IiwibGFuZ3VhZ2UiOiJoaSIsImtpbmQiOiJjb21wb3NpdGlvbiIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IldoYXQgcGF5bWVudCB0eXBlcyBhcmUgYWNjZXB0ZWQgYnkgdGhlIG9yZ2FuaXplciBvZiB0aGUgZXZlbnQgW0VOVElUWV9BXSBwYXJ0aWNpcGF0ZWQgaW4/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IuCknOCkv+CkuCDgpIbgpK/gpYvgpJzgpKgg4KSu4KWH4KSCIFtFTlRJVFlfQV0g4KSo4KWHIOCkreCkvuCklyDgpLLgpL/gpK/gpL4g4KSl4KS+LCDgpIngpLjgpJXgpL4g4KSG4KSv4KWL4KSc4KSVIOCkleCkv+CkqCDgpK3gpYHgpJfgpKTgpL7gpKgg4KS14KS/4KSn4KS/4KSv4KWL4KSCIOCkleCliyDgpLjgpY3gpLXgpYDgpJXgpL7gpLAg4KSV4KSw4KSk4KS+IOCkueCliD8ifSx7InJldmlld19rZXkiOiJSMDQ5MyIsImxhbmd1YWdlIjoiaGkiLCJraW5kIjoiY29tcG9zaXRpb24iLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJXaGljaCBpbnN0aXR1dGlvbiBob2xkcyB0aGUgZG9jdW1lbnRhdGlvbiBmaWxlcyBhYm91dCB0aGUgcGVyc29uIGFmdGVyIHdob20gdGhlIGNpdHkgdGhhdCBob3VzZXMgdGhlIFtFTlRJVFlfQV0ncyBoZWFkcXVhcnRlcnMgaXMgbmFtZWQ/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IuCknOCkv+CkuCDgpLbgpLngpLAg4KSu4KWH4KSCIFtFTlRJVFlfQV0g4KSV4KS+IOCkruClgeCkluCljeCkr+CkvuCksuCkryDgpLjgpY3gpKXgpL/gpKQg4KS54KWILCDgpIngpLgg4KS24KS54KSwIOCkleCkviDgpKjgpL7gpK4g4KSc4KS/4KS4IOCkteCljeCkr+CkleCljeCkpOCkvyDgpJXgpYcg4KSo4KS+4KSuIOCkquCksCDgpLDgpJbgpL4g4KSX4KSv4KS+IOCkueCliCwg4KSJ4KS44KS44KWHIOCkuOCkguCkrOCkguCkp+Ckv+CkpCDgpKbgpLjgpY3gpKTgpL7gpLXgpYfgpJzgpLzgpYvgpIIg4KSV4KWAIOCkq+CkvuCkh+CksuClh+CkgiDgpJXgpL/gpLgg4KS44KSC4KS44KWN4KSl4KS+4KSoIOCkleClhyDgpKrgpL7gpLgg4KS54KWI4KSCPyJ9LHsicmV2aWV3X2tleSI6IlIwNTIxIiwibGFuZ3VhZ2UiOiJoaSIsImtpbmQiOiJjb21wb3NpdGlvbiIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IldoZXJlIGlzIHRoZSBlbnRpdHkgdGhhdCBib3JkZXJzIHRoZSBwbGFjZSBbRU5USVRZX0FdIGlzIGluZGlnZW5vdXMgdG8gaGVhZHF1YXJ0ZXJlZD8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoi4KSc4KS/4KS4IOCkuOCljeCkpeCkvuCkqCDgpJXgpL4gW0VOVElUWV9BXSDgpK7gpYLgpLIg4KSo4KS/4KS14KS+4KS44KWAIOCkueCliCwg4KSJ4KS44KSV4KWAIOCkuOClgOCkruCkviDgpLjgpYcg4KSy4KSX4KWAIOCkh+CkleCkvuCkiCDgpJXgpL4g4KSu4KWB4KSW4KWN4KSv4KS+4KSy4KSvIOCkleCkueCkvuCkgSDgpLjgpY3gpKXgpL/gpKQg4KS54KWIPyJ9LHsicmV2aWV3X2tleSI6IlIwNTQ0IiwibGFuZ3VhZ2UiOiJ0aCIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJJbiB3aGF0IHllYXIgZGlkIFtFTlRJVFlfQV0gZm91bmQgW0VOVElUWV9CXT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiW0VOVElUWV9BXSDguIHguYjguK3guJXguLHguYnguIcgW0VOVElUWV9CXSDguYPguJnguJvguLXguYPguJQifSx7InJldmlld19rZXkiOiJSMDU1MyIsImxhbmd1YWdlIjoidGgiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiV2hlbiBkaWQgW0VOVElUWV9BXSBiZWNvbWUgcGFydCBvZiB0aGUgW0VOVElUWV9CXT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiW0VOVElUWV9BXSDguIHguKXguLLguKLguYDguJvguYfguJnguKrguYjguKfguJnguKvguJnguLbguYjguIfguILguK3guIcgW0VOVElUWV9CXSDguYDguKHguLfguYjguK3guYPguJQifSx7InJldmlld19rZXkiOiJSMDU3MSIsImxhbmd1YWdlIjoidGgiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiT24gd2hhdCBkYXRlIGRpZCBbRU5USVRZX0FdIGJlY29tZSB0aGUgY291bnRyeSB0aGF0IFtFTlRJVFlfQl0gaXMgaW4/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IltFTlRJVFlfQV0g4LiB4Lil4Liy4Lii4LmA4Lib4LmH4LiZ4Lib4Lij4Liw4LmA4LiX4Lio4LiX4Li14LmIIFtFTlRJVFlfQl0g4LiV4Lix4LmJ4LiH4Lit4Lii4Li54LmI4LmA4Lih4Li34LmI4Lit4Lin4Lix4LiZ4LiX4Li14LmI4LmA4LiX4LmI4Liy4LmD4LiUIn0seyJyZXZpZXdfa2V5IjoiUjA1ODciLCJsYW5ndWFnZSI6InRoIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IkluIHdoYXQgeWVhciB3YXMgW0VOVElUWV9BXSBub21pbmF0ZWQgZm9yIFtFTlRJVFlfQl0/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IltFTlRJVFlfQV0g4LmE4LiU4LmJ4Lij4Lix4Lia4LiB4Liy4Lij4LmA4Liq4LiZ4Lit4LiK4Li34LmI4Lit4LmA4LiC4LmJ4Liy4LiK4Li04LiHIFtFTlRJVFlfQl0g4LmD4LiZ4Lib4Li14LmD4LiUIn0seyJyZXZpZXdfa2V5IjoiUjA2MDIiLCJsYW5ndWFnZSI6InRoIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6Ik9uIHdoYXQgZGF0ZSBkaWQgW0VOVElUWV9BXSBjZWFzZSB0byBiZSB0aGUgY291bnRyeSBvZiBbRU5USVRZX0JdPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJbRU5USVRZX0FdIOC4quC4tOC5ieC4meC4quC4uOC4lOC4geC4suC4o+C5gOC4m+C5h+C4meC4m+C4o+C4sOC5gOC4l+C4qOC4l+C4teC5iCBbRU5USVRZX0JdIOC4leC4seC5ieC4h+C4reC4ouC4ueC5iOC5gOC4oeC4t+C5iOC4reC4p+C4seC4meC4l+C4teC5iOC5gOC4l+C5iOC4suC5g+C4lCJ9LHsicmV2aWV3X2tleSI6IlIwNjc3IiwibGFuZ3VhZ2UiOiJ0aCIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJJbiB3aGF0IHllYXIgZGlkIFtFTlRJVFlfQV0gcmVjZWl2ZSB0aGUgW0VOVElUWV9CXT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiW0VOVElUWV9BXSDguYTguJTguYnguKPguLHguJogW0VOVElUWV9CXSDguYPguJnguJvguLXguYPguJQifV0="
INPUT_SHA256 = "2093e8eab1bbd56e8f939a96034bf50745f409072c359d12a8520ad2d2d9c961"
OUT = Path("/kaggle/working/alias_probe.json")

def sha(data):
    return hashlib.sha256(data).hexdigest()

def mask(text):
    expected = collections.Counter()
    originals = {}
    def sub(match):
        marker = match.group(0)
        alias = ALIASES.get(marker)
        if alias is None:
            raise ValueError(f"No approved natural alias for {marker}")
        if alias.lower() in text.lower().replace(marker.lower(), ""):
            raise ValueError(f"Alias collision for {marker}")
        expected[alias] += 1
        originals[alias] = marker
        return alias
    masked = re.sub(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]", sub, text)
    if not expected:
        raise ValueError("No protected placeholder in candidate")
    return masked, originals, expected

def restore(text, originals, expected):
    found = collections.Counter({alias: text.count(alias) for alias in originals})
    if found != expected:
        raise ValueError(f"alias count changed: expected={dict(expected)}, found={dict(found)}")
    for alias, marker in originals.items():
        text = text.replace(alias, marker)
    if any(alias in text for alias in originals):
        raise ValueError("an alias was not restored")
    return text.strip()

def main():
    raw_input = base64.b64decode(RAW_INPUT_B64)
    if sha(raw_input) != INPUT_SHA256:
        raise ValueError("embedded-input hash mismatch")
    rows = json.loads(raw_input)
    if len(rows) != 28 or len({row["review_key"] for row in rows}) != 28:
        raise ValueError("expected 28 unique, source-bound blocked rows")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; refusing to run a different-device probe")
    runtime = {"status": "RUNNING_MARKER_SURVIVAL_ONLY", "rows": [],
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "input_sha256": INPUT_SHA256, "torch": torch.__version__,
        "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "translation_quality_reviewed": False, "answers_or_splits_sent": False}
    tok = NllbTokenizerFast.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
        torch_dtype=torch.float16, device_map="auto")
    model.eval()
    for i, row in enumerate(rows, 1):
        result = {"review_key": row["review_key"], "language": row["language"],
            "kind": row["kind"], "source_sha256": sha(row["source_question_en"].encode()),
            "translation_sha256": sha(row["translated_question"].encode()),
            "alias_counts": {}, "backtranslation_raw": "", "restored_backtranslation_en": "",
            "status": "BLOCKED"}
        try:
            src_markers = collections.Counter(re.findall(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]", row["source_question_en"]))
            tgt_markers = collections.Counter(re.findall(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]", row["translated_question"]))
            if src_markers != tgt_markers:
                raise ValueError(f"source/target marker mismatch: {dict(src_markers)} != {dict(tgt_markers)}")
            protected, originals, expected = mask(row["translated_question"])
            result["alias_counts"] = dict(expected)
            tok.src_lang = NLLB_CODES[row["language"]]
            encoded = tok(protected, return_tensors="pt", truncation=True, max_length=256).to(model.device)
            with torch.inference_mode():
                output = model.generate(**encoded,
                    forced_bos_token_id=tok.convert_tokens_to_ids("eng_Latn"),
                    max_new_tokens=128, do_sample=False)
            translated = tok.batch_decode(output, skip_special_tokens=True)[0]
            result["backtranslation_raw"] = translated
            restored = restore(translated, originals, expected)
            if collections.Counter(re.findall(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]", restored)) != src_markers:
                raise ValueError("restored marker multiset does not match English source")
            result["restored_backtranslation_en"] = restored
            result["status"] = "PASS_ALIAS_COUNT_ONLY"
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        runtime["rows"].append(result)
        print(f"Processed {i}/{len(rows)}: {result['status']}", flush=True)
    runtime["counts"] = dict(collections.Counter(r["status"] for r in runtime["rows"]))
    runtime["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    runtime["status"] = "DIAGNOSTIC_COMPLETE_NOT_TRANSLATION_APPROVAL"
    OUT.write_text(json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": runtime["status"], "counts": runtime["counts"],
        "output": str(OUT)}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()

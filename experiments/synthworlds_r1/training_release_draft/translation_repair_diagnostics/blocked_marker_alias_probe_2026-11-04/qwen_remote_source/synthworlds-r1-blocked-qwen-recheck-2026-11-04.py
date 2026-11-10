#!/usr/bin/env python3
"""Blinded, one-row Qwen semantic/fluency recheck with raw-output provenance."""
import base64, collections, hashlib, json, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.51.3", "accelerate==1.15.0", "bitsandbytes==0.50.2"], check=True)
import torch, transformers, accelerate, bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
INPUT_SHA256 = "a6f92f97f954ab8392133d2c6501753c8d2dbdb63f2fc8582451de03418c1658"
RAW_INPUT_B64 = "W3sicmV2aWV3X2tleSI6IlIwMDExIiwibGFuZ3VhZ2UiOiJmciIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJPbiB3aGF0IGRhdGUgZGlkIFtFTlRJVFlfQV0gYmVjb21lIGEgcmVsYXRpdmUgb2YgW0VOVElUWV9CXT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiw4AgcXVlbGxlIGRhdGUgW0VOVElUWV9BXSBlc3QtaWwgZGV2ZW51IHVuIG1lbWJyZSBkZSBsYSBmYW1pbGxlIGRlIFtFTlRJVFlfQl0gPyIsImJhY2tfdHJhbnNsYXRpb25fZW4iOiJXaGVuIGRpZCBbRU5USVRZX0FdIGJlY29tZSBhIG1lbWJlciBvZiBbRU5USVRZX0JdJ3MgZmFtaWx5PyJ9LHsicmV2aWV3X2tleSI6IlIwMDM2IiwibGFuZ3VhZ2UiOiJmciIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJJbiB3aGF0IHllYXIgd2FzIFtFTlRJVFlfQV0gdGhlIGFyY2hpdGVjdCBvZiBbRU5USVRZX0JdPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJFbiBxdWVsbGUgYW5uw6llIFtFTlRJVFlfQV0gw6l0YWl0LWlsIGzigJlhcmNoaXRlY3RlIGRlIFtFTlRJVFlfQl0gPyIsImJhY2tfdHJhbnNsYXRpb25fZW4iOiJXaGF0IHllYXIgd2FzIFtFTlRJVFlfQV0gW0VOVElUWV9CXSdzIGFyY2hpdGVjdD8ifSx7InJldmlld19rZXkiOiJSMDE3OSIsImxhbmd1YWdlIjoiZGUiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiT24gd2hhdCBkYXRlIGRpZCBbRU5USVRZX0FdIGJlY29tZSB0aGUgY291bnRyeSB0aGF0IFtFTlRJVFlfQl0gaXMgaW4/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IkFuIHdlbGNoZW0gRGF0dW0gd3VyZGUgW0VOVElUWV9BXSB6dSBkZW0gTGFuZCwgaW4gZGVtIHNpY2ggW0VOVElUWV9CXSBiZWZpbmRldD8iLCJiYWNrX3RyYW5zbGF0aW9uX2VuIjoiV2hlbiBkaWQgW0VOVElUWV9BXSBiZWNvbWUgdGhlIGNvdW50cnkgd2hlcmUgW0VOVElUWV9CXSBpcz8ifSx7InJldmlld19rZXkiOiJSMDIwNiIsImxhbmd1YWdlIjoiZGUiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiV2hhdCBpcyB0aGUgW0VOVElUWV9BXSBkaWZmZXJlbnQgZnJvbT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiV292b24gdW50ZXJzY2hlaWRldCBzaWNoIFtFTlRJVFlfQV0/IiwiYmFja190cmFuc2xhdGlvbl9lbiI6IldoYXQncyBbRU5USVRZX0FdIGRpZmZlcmVudCBhYm91dD8ifSx7InJldmlld19rZXkiOiJSMDIyMiIsImxhbmd1YWdlIjoiZGUiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiV2hpY2ggZW50aXR5IGlzIFtFTlRJVFlfQV0gZGlmZmVyZW50IGZyb20/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IlZvbiB3ZWxjaGVyIEVudGl0w6R0IHVudGVyc2NoZWlkZXQgc2ljaCBbRU5USVRZX0FdPyIsImJhY2tfdHJhbnNsYXRpb25fZW4iOiJXaGF0IGVudGl0eSBpcyBbRU5USVRZX0FdIGRpZmZlcmVudCBmcm9tPyJ9LHsicmV2aWV3X2tleSI6IlIwMjM4IiwibGFuZ3VhZ2UiOiJkZSIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJPbiB3aGF0IGRhdGUgZGlkIFtFTlRJVFlfQV0gc3RvcCBiZWluZyB0aGUgY2FwaXRhbCBvZiBbRU5USVRZX0JdPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJBbiB3ZWxjaGVtIERhdHVtIGjDtnJ0ZSBbRU5USVRZX0FdIGF1ZiwgZGllIEhhdXB0c3RhZHQgdm9uIFtFTlRJVFlfQl0genUgc2Vpbj8iLCJiYWNrX3RyYW5zbGF0aW9uX2VuIjoiV2hlbiBkaWQgW0VOVElUWV9BXSBzdG9wIGJlaW5nIFtFTlRJVFlfQl0ncyBjYXBpdGFsPyJ9LHsicmV2aWV3X2tleSI6IlIwMzQ0IiwibGFuZ3VhZ2UiOiJoaSIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJJbiB3aGF0IHllYXIgZGlkIFtFTlRJVFlfQV0gZm91bmQgW0VOVElUWV9CXT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiW0VOVElUWV9BXSDgpKjgpYcgW0VOVElUWV9CXSDgpJXgpYAg4KS44KWN4KSl4KS+4KSq4KSo4KS+IOCkleCkv+CkuCDgpLXgpLDgpY3gpLcg4KSV4KWAIOCkpeClgD8iLCJiYWNrX3RyYW5zbGF0aW9uX2VuIjoiV2hhdCB5ZWFyIGRpZCBbRU5USVRZX0FdIGZvdW5kIFtFTlRJVFlfQl0/In0seyJyZXZpZXdfa2V5IjoiUjAzNDkiLCJsYW5ndWFnZSI6ImhpIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6Ik9uIHdoYXQgZGF0ZSBkaWQgW0VOVElUWV9BXSBiZWNvbWUgYSByZWxhdGl2ZSBvZiBbRU5USVRZX0JdPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJbRU5USVRZX0FdIOCkleCkv+CkuCDgpKTgpL7gpLDgpYDgpJYg4KSV4KWLIFtFTlRJVFlfQl0g4KSV4KS+IOCksOCkv+CktuCljeCkpOClh+CkpuCkvuCksCDgpKzgpKjgpL4/IiwiYmFja190cmFuc2xhdGlvbl9lbiI6Ik9uIHdoYXQgZGF0ZSBkaWQgW0VOVElUWV9BXSBiZWNvbWUgW0VOVElUWV9CXSdzIHJlbGF0aXZlPyJ9LHsicmV2aWV3X2tleSI6IlIwMzYwIiwibGFuZ3VhZ2UiOiJoaSIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJPbiB3aGF0IGRhdGUgZGlkIFtFTlRJVFlfQV0gYmVjb21lIHRoZSBjb3VudHJ5IHRoYXQgW0VOVElUWV9CXSBpcyBpbj8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoi4KSV4KS/4KS4IOCkpOCkvuCksOClgOCkluCkvCDgpJXgpYsgW0VOVElUWV9BXSDgpLXgpLkg4KSm4KWH4KS2IOCkrOCkqOCkviDgpJzgpL/gpLjgpK7gpYfgpIIgW0VOVElUWV9CXSDgpLjgpY3gpKXgpL/gpKQg4KS54KWIPyIsImJhY2tfdHJhbnNsYXRpb25fZW4iOiJPbiB3aGF0IGRhdGUgZGlkIFtFTlRJVFlfQV0gY3JlYXRlIHRoZSBjb3VudHJ5IHdoZXJlIFtFTlRJVFlfQl0gaXMgbG9jYXRlZD8ifSx7InJldmlld19rZXkiOiJSMDM3MSIsImxhbmd1YWdlIjoiaGkiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiSW4gd2hhdCB5ZWFyIHdhcyBbRU5USVRZX0FdIG5vbWluYXRlZCBmb3IgW0VOVElUWV9CXT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiW0VOVElUWV9BXSDgpJXgpYsgW0VOVElUWV9CXSDgpJXgpYcg4KSy4KS/4KSPIOCkleCkv+CkuCDgpLXgpLDgpY3gpLcg4KSo4KS+4KSu4KS+4KSC4KSV4KS/4KSkIOCkleCkv+Ckr+CkviDgpJfgpK/gpL4g4KSl4KS+PyIsImJhY2tfdHJhbnNsYXRpb25fZW4iOiJXaGF0IHllYXIgd2FzIFtFTlRJVFlfQV0gbm9taW5hdGVkIGZvciBbRU5USVRZX0JdPyJ9LHsicmV2aWV3X2tleSI6IlIwMzc5IiwibGFuZ3VhZ2UiOiJoaSIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJPbiB3aGF0IGRhdGUgZGlkIFtFTlRJVFlfQV0gY2Vhc2UgdG8gYmUgdGhlIGNvdW50cnkgb2YgW0VOVElUWV9CXT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoi4KSV4KS/4KS4IOCkpOCkvuCksOClgOCkluCkvCDgpJXgpYsgW0VOVElUWV9BXSwgW0VOVElUWV9CXSDgpJXgpL4g4KSm4KWH4KS2IOCkqOCkueClgOCkgiDgpLDgpLngpL4/IiwiYmFja190cmFuc2xhdGlvbl9lbiI6IldoYXQgZGF0ZSBkaWQgW0VOVElUWV9BXSwgW0VOVElUWV9CXSwgbm90IGdvIHRvPyJ9LHsicmV2aWV3X2tleSI6IlIwNDA3IiwibGFuZ3VhZ2UiOiJoaSIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJJbiB3aGF0IHllYXIgZGlkIFtFTlRJVFlfQV0gYmVjb21lIHRoZSBzcG91c2Ugb2YgW0VOVElUWV9CXT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiW0VOVElUWV9BXSDgpJXgpL/gpLgg4KS14KSw4KWN4KS3IFtFTlRJVFlfQl0g4KSV4KS+IOCknOClgOCkteCkqOCkuOCkvuCkpeClgCDgpKzgpKjgpL4/IiwiYmFja190cmFuc2xhdGlvbl9lbiI6IkluIHdoYXQgeWVhciBkaWQgW0VOVElUWV9BXSBiZWNvbWUgW0VOVElUWV9CXSdzIHdpZmU/In0seyJyZXZpZXdfa2V5IjoiUjA0MTciLCJsYW5ndWFnZSI6ImhpIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6Ik9uIHdoYXQgZGF0ZSBkaWQgW0VOVElUWV9BXSBzdG9wIGJlaW5nIHRoZSBjYXBpdGFsIG9mIFtFTlRJVFlfQl0/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IuCkleCkv+CkuCDgpKTgpL7gpLDgpYDgpJbgpLwg4KSV4KWLIFtFTlRJVFlfQV0sIFtFTlRJVFlfQl0g4KSV4KWAIOCksOCkvuCknOCkp+CkvuCkqOClgCDgpKjgpLngpYDgpIIg4KSw4KS54KS+PyIsImJhY2tfdHJhbnNsYXRpb25fZW4iOiJXaGF0IGRhdGUgd2FzIFtFTlRJVFlfQV0sIFtFTlRJVFlfQl0ncyBjYXBpdGFsIG5vdD8ifSx7InJldmlld19rZXkiOiJSMDQxOCIsImxhbmd1YWdlIjoiaGkiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiSW4gd2hhdCB5ZWFyIGRpZCBbRU5USVRZX0FdJ3MgdGVybSBhcyBoZWFkIG9mIGdvdmVybm1lbnQgb2YgW0VOVElUWV9CXSBlbmQ/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IltFTlRJVFlfQl0g4KSV4KWHIOCktuCkvuCkuOCkqOCkvuCkp+CljeCkr+CkleCljeCktyDgpJXgpYcg4KSw4KWC4KSqIOCkruClh+CkgiBbRU5USVRZX0FdIOCkleCkviDgpJXgpL7gpLDgpY3gpK/gpJXgpL7gpLIg4KSV4KS/4KS4IOCkteCksOCljeCktyDgpLjgpK7gpL7gpKrgpY3gpKQg4KS54KWB4KSGPyIsImJhY2tfdHJhbnNsYXRpb25fZW4iOiJJbiB3aGF0IHllYXIgZGlkIFtFTlRJVFlfQV0ncyB0ZXJtIGFzIFtFTlRJVFlfQl0ncyBnb3Zlcm5vciBlbmQ/In0seyJyZXZpZXdfa2V5IjoiUjA0MjEiLCJsYW5ndWFnZSI6ImhpIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IldoYXQgaW1tZWRpYXRlbHkgZm9sbG93cyBbRU5USVRZX0FdPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJbRU5USVRZX0FdIOCkleClhyDgpKTgpYHgpLDgpILgpKQg4KSs4KS+4KSmIOCkleCljeCkr+CkviDgpIbgpKTgpL4g4KS54KWIPyIsImJhY2tfdHJhbnNsYXRpb25fZW4iOiJXaGF0IGNvbWVzIGFmdGVyIFtFTlRJVFlfQV0/In0seyJyZXZpZXdfa2V5IjoiUjA0MzYiLCJsYW5ndWFnZSI6ImhpIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IldoZW4gZGlkIFtFTlRJVFlfQV0gYmVjb21lIHBhcnQgb2YgdGhlIFtFTlRJVFlfQl0/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IltFTlRJVFlfQV0g4KSV4KSsIFtFTlRJVFlfQl0g4KSV4KS+IOCkueCkv+CkuOCljeCkuOCkviDgpKzgpKjgpL4/IiwiYmFja190cmFuc2xhdGlvbl9lbiI6IldoZW4gZGlkIFtFTlRJVFlfQV0gYmVjb21lIHBhcnQgb2YgW0VOVElUWV9CXT8ifSx7InJldmlld19rZXkiOiJSMDQ0MiIsImxhbmd1YWdlIjoiaGkiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiSW4gd2hhdCB5ZWFyIGRpZCBbRU5USVRZX0FdIHJlY2VpdmUgdGhlIFtFTlRJVFlfQl0/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IltFTlRJVFlfQV0g4KSV4KWLIFtFTlRJVFlfQl0g4KSV4KS/4KS4IOCkteCksOCljeCktyDgpKrgpY3gpLDgpL7gpKrgpY3gpKQg4KS54KWB4KSGIOCkpeCkvj8iLCJiYWNrX3RyYW5zbGF0aW9uX2VuIjoiV2hhdCB5ZWFyIGRpZCBbRU5USVRZX0FdIHJlY2VpdmUgW0VOVElUWV9CXT8ifSx7InJldmlld19rZXkiOiJSMDQ0OSIsImxhbmd1YWdlIjoiaGkiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiSW4gd2hhdCB5ZWFyIGRpZCBbRU5USVRZX0FdIHN0b3Agc2VydmluZyBhcyBhIGJvYXJkIG1lbWJlciBvZiBbRU5USVRZX0JdPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJbRU5USVRZX0FdIOCkqOClhyDgpJXgpL/gpLgg4KS14KSw4KWN4KS3IFtFTlRJVFlfQl0g4KSV4KWHIOCkrOCli+CksOCljeCkoSDgpLjgpKbgpLjgpY3gpK8g4KSV4KWHIOCksOClguCkqiDgpK7gpYfgpIIg4KSV4KS+4KSw4KWN4KSvIOCkleCksOCkqOCkviDgpKzgpILgpKYg4KSV4KS/4KSv4KS+PyIsImJhY2tfdHJhbnNsYXRpb25fZW4iOiJJbiB3aGF0IHllYXIgZGlkIFtFTlRJVFlfQV0gY2Vhc2UgdG8gc2VydmUgYXMgYSBib2FyZCBtZW1iZXIgb2YgW0VOVElUWV9CXT8ifSx7InJldmlld19rZXkiOiJSMDQ1NSIsImxhbmd1YWdlIjoiaGkiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiSW4gd2hhdCB5ZWFyIHdhcyBbRU5USVRZX0FdIGEgZG9jdG9yYWwgc3R1ZGVudCBvZiBbRU5USVRZX0JdPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJbRU5USVRZX0FdIOCkleCkv+CkuCDgpLXgpLDgpY3gpLcgW0VOVElUWV9CXSDgpJXgpL4g4KSh4KWJ4KSV4KWN4KSf4KSw4KWH4KSfIOCkm+CkvuCkpOCljeCksCDgpKXgpL4/IiwiYmFja190cmFuc2xhdGlvbl9lbiI6IldoYXQgeWVhciB3YXMgW0VOVElUWV9BXSBbRU5USVRZX0JdJ3MgZG9jdG9yYWwgc3R1ZGVudD8ifSx7InJldmlld19rZXkiOiJSMDQ4MSIsImxhbmd1YWdlIjoiaGkiLCJraW5kIjoiY29tcG9zaXRpb24iLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJXaGljaCBwb2xpdGljYWwgb2ZmaWNlIG9mIHRoZSBzdGF0ZSB0aGF0IHJlcGxhY2VkIHRoZSBjb3VudHJ5IHdob3NlIGNhcGl0YWwgaXMgW0VOVElUWV9BXSBpcyBoZWxkIGJ5IGl0cyBoZWFkIG9mIHN0YXRlPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiLgpJzgpL/gpLgg4KSm4KWH4KS2IOCkleClgCDgpLDgpL7gpJzgpKfgpL7gpKjgpYAgW0VOVElUWV9BXSDgpLngpYgsIOCkieCkuOCkleCkviDgpLjgpY3gpKXgpL7gpKgg4KSy4KWH4KSo4KWHIOCkteCkvuCksuClhyDgpLDgpL7gpJzgpY3gpK8g4KSV4KWHIOCksOCkvuCkt+CljeCkn+CljeCksOCkvuCkp+CljeCkr+CkleCljeCktyDgpJXgpYcg4KSq4KS+4KS4IOCkieCkuCDgpLDgpL7gpJzgpY3gpK8g4KSV4KS+IOCkleCljOCkqC3gpLjgpL4g4KSw4KS+4KSc4KSo4KWA4KSk4KS/4KSVIOCkquCkpiDgpLngpYg/IiwiYmFja190cmFuc2xhdGlvbl9lbiI6IldoYXQgcG9saXRpY2FsIHBvc2l0aW9uIGRvZXMgdGhlIGhlYWQgb2Ygc3RhdGUgb2YgdGhlIGNvdW50cnkgd2hlcmUgW0VOVElUWV9BXSBpcyB0aGUgY2FwaXRhbCBoYXZlPyJ9LHsicmV2aWV3X2tleSI6IlIwNDg5IiwibGFuZ3VhZ2UiOiJoaSIsImtpbmQiOiJjb21wb3NpdGlvbiIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IldoYXQgcGF5bWVudCB0eXBlcyBhcmUgYWNjZXB0ZWQgYnkgdGhlIG9yZ2FuaXplciBvZiB0aGUgZXZlbnQgW0VOVElUWV9BXSBwYXJ0aWNpcGF0ZWQgaW4/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IuCknOCkv+CkuCDgpIbgpK/gpYvgpJzgpKgg4KSu4KWH4KSCIFtFTlRJVFlfQV0g4KSo4KWHIOCkreCkvuCklyDgpLLgpL/gpK/gpL4g4KSl4KS+LCDgpIngpLjgpJXgpL4g4KSG4KSv4KWL4KSc4KSVIOCkleCkv+CkqCDgpK3gpYHgpJfgpKTgpL7gpKgg4KS14KS/4KSn4KS/4KSv4KWL4KSCIOCkleCliyDgpLjgpY3gpLXgpYDgpJXgpL7gpLAg4KSV4KSw4KSk4KS+IOCkueCliD8iLCJiYWNrX3RyYW5zbGF0aW9uX2VuIjoiV2hhdCBwYXltZW50IG1ldGhvZHMgZG9lcyB0aGUgb3JnYW5pemVyIG9mIHRoZSBldmVudCBpbiB3aGljaCBbRU5USVRZX0FdIHBhcnRpY2lwYXRlZCBhY2NlcHQ/In0seyJyZXZpZXdfa2V5IjoiUjA0OTMiLCJsYW5ndWFnZSI6ImhpIiwia2luZCI6ImNvbXBvc2l0aW9uIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiV2hpY2ggaW5zdGl0dXRpb24gaG9sZHMgdGhlIGRvY3VtZW50YXRpb24gZmlsZXMgYWJvdXQgdGhlIHBlcnNvbiBhZnRlciB3aG9tIHRoZSBjaXR5IHRoYXQgaG91c2VzIHRoZSBbRU5USVRZX0FdJ3MgaGVhZHF1YXJ0ZXJzIGlzIG5hbWVkPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiLgpJzgpL/gpLgg4KS24KS54KSwIOCkruClh+CkgiBbRU5USVRZX0FdIOCkleCkviDgpK7gpYHgpJbgpY3gpK/gpL7gpLLgpK8g4KS44KWN4KSl4KS/4KSkIOCkueCliCwg4KSJ4KS4IOCktuCkueCksCDgpJXgpL4g4KSo4KS+4KSuIOCknOCkv+CkuCDgpLXgpY3gpK/gpJXgpY3gpKTgpL8g4KSV4KWHIOCkqOCkvuCkriDgpKrgpLAg4KSw4KSW4KS+IOCkl+Ckr+CkviDgpLngpYgsIOCkieCkuOCkuOClhyDgpLjgpILgpKzgpILgpKfgpL/gpKQg4KSm4KS44KWN4KSk4KS+4KS14KWH4KSc4KS84KWL4KSCIOCkleClgCDgpKvgpL7gpIfgpLLgpYfgpIIg4KSV4KS/4KS4IOCkuOCkguCkuOCljeCkpeCkvuCkqCDgpJXgpYcg4KSq4KS+4KS4IOCkueCliOCkgj8iLCJiYWNrX3RyYW5zbGF0aW9uX2VuIjoiV2hpY2ggaW5zdGl0dXRpb24gaGFzIGZpbGVzIG9mIGRvY3VtZW50cyByZWxhdGluZyB0byB0aGUgcGVyc29uIG5hbWVkIGFmdGVyIHRoZSBjaXR5IGluIHdoaWNoIFtFTlRJVFlfQV0ncyBoZWFkcXVhcnRlcnMgaXMgbG9jYXRlZD8ifSx7InJldmlld19rZXkiOiJSMDUxMiIsImxhbmd1YWdlIjoiaGkiLCJraW5kIjoiY29tcG9zaXRpb24iLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJXaGF0IGltbWVkaWF0ZWx5IGZvbGxvd3MgW0VOVElUWV9BXT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiW0VOVElUWV9BXSDgpJXgpYcg4KSk4KWB4KSw4KSC4KSkIOCkrOCkvuCkpiDgpJXgpY3gpK/gpL4g4KSG4KSk4KS+IOCkueCliD8iLCJiYWNrX3RyYW5zbGF0aW9uX2VuIjoiV2hhdCBjb21lcyBhZnRlciBbRU5USVRZX0FdPyJ9LHsicmV2aWV3X2tleSI6IlIwNTIxIiwibGFuZ3VhZ2UiOiJoaSIsImtpbmQiOiJjb21wb3NpdGlvbiIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IldoZXJlIGlzIHRoZSBlbnRpdHkgdGhhdCBib3JkZXJzIHRoZSBwbGFjZSBbRU5USVRZX0FdIGlzIGluZGlnZW5vdXMgdG8gaGVhZHF1YXJ0ZXJlZD8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoi4KSc4KS/4KS4IOCkuOCljeCkpeCkvuCkqCDgpJXgpL4gW0VOVElUWV9BXSDgpK7gpYLgpLIg4KSo4KS/4KS14KS+4KS44KWAIOCkueCliCwg4KSJ4KS44KSV4KWAIOCkuOClgOCkruCkviDgpLjgpYcg4KSy4KSX4KWAIOCkh+CkleCkvuCkiCDgpJXgpL4g4KSu4KWB4KSW4KWN4KSv4KS+4KSy4KSvIOCkleCkueCkvuCkgSDgpLjgpY3gpKXgpL/gpKQg4KS54KWIPyIsImJhY2tfdHJhbnNsYXRpb25fZW4iOiJXaGVyZSBpcyB0aGUgaGVhZHF1YXJ0ZXJzIG9mIHRoZSB1bml0IGxvY2F0ZWQgb24gdGhlIGJvcmRlciBvZiB0aGUgcGxhY2Ugd2hlcmUgW0VOVElUWV9BXSBpcyBuYXRpdmU/In0seyJyZXZpZXdfa2V5IjoiUjA1MzciLCJsYW5ndWFnZSI6ImhpIiwia2luZCI6ImNvbXBvc2l0aW9uIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiV2hpY2ggbm9ibGUgdGl0bGUgZG9lcyBbRU5USVRZX0FdIGhvbGQ/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IltFTlRJVFlfQV0g4KSV4KWHIOCkquCkvuCkuCDgpJXgpYzgpKgt4KS44KWAIOCkleClgeCksuClgOCkqCDgpIngpKrgpL7gpKfgpL8g4KS54KWIPyIsImJhY2tfdHJhbnNsYXRpb25fZW4iOiJXaGF0IGlzIHRoZSB0aXRsZSBvZiB0aGUgW0VOVElUWV9BXT8ifSx7InJldmlld19rZXkiOiJSMDU0NCIsImxhbmd1YWdlIjoidGgiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiSW4gd2hhdCB5ZWFyIGRpZCBbRU5USVRZX0FdIGZvdW5kIFtFTlRJVFlfQl0/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IltFTlRJVFlfQV0g4LiB4LmI4Lit4LiV4Lix4LmJ4LiHIFtFTlRJVFlfQl0g4LmD4LiZ4Lib4Li14LmD4LiUIiwiYmFja190cmFuc2xhdGlvbl9lbiI6IltFTlRJVFlfQV0gZm91bmRlZCBbRU5USVRZX0JdIGluIHdoYXQgeWVhcj8ifSx7InJldmlld19rZXkiOiJSMDU1MyIsImxhbmd1YWdlIjoidGgiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiV2hlbiBkaWQgW0VOVElUWV9BXSBiZWNvbWUgcGFydCBvZiB0aGUgW0VOVElUWV9CXT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiW0VOVElUWV9BXSDguIHguKXguLLguKLguYDguJvguYfguJnguKrguYjguKfguJnguKvguJnguLbguYjguIfguILguK3guIcgW0VOVElUWV9CXSDguYDguKHguLfguYjguK3guYPguJQiLCJiYWNrX3RyYW5zbGF0aW9uX2VuIjoiW0VOVElUWV9BXSBiZWNhbWUgcGFydCBvZiBbRU5USVRZX0JdIHdoZW4uIn0seyJyZXZpZXdfa2V5IjoiUjA1NzEiLCJsYW5ndWFnZSI6InRoIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6Ik9uIHdoYXQgZGF0ZSBkaWQgW0VOVElUWV9BXSBiZWNvbWUgdGhlIGNvdW50cnkgdGhhdCBbRU5USVRZX0JdIGlzIGluPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJbRU5USVRZX0FdIOC4geC4peC4suC4ouC5gOC4m+C5h+C4meC4m+C4o+C4sOC5gOC4l+C4qOC4l+C4teC5iCBbRU5USVRZX0JdIOC4leC4seC5ieC4h+C4reC4ouC4ueC5iOC5gOC4oeC4t+C5iOC4reC4p+C4seC4meC4l+C4teC5iOC5gOC4l+C5iOC4suC5g+C4lCIsImJhY2tfdHJhbnNsYXRpb25fZW4iOiJbRU5USVRZX0FdIGJlY2FtZSB0aGUgY291bnRyeSB3aGVyZSBbRU5USVRZX0JdIHdhcyBsb2NhdGVkIG9uIHdoYXQgZGF0ZT8ifSx7InJldmlld19rZXkiOiJSMDU4NyIsImxhbmd1YWdlIjoidGgiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiSW4gd2hhdCB5ZWFyIHdhcyBbRU5USVRZX0FdIG5vbWluYXRlZCBmb3IgW0VOVElUWV9CXT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiW0VOVElUWV9BXSDguYTguJTguYnguKPguLHguJrguIHguLLguKPguYDguKrguJnguK3guIrguLfguYjguK3guYDguILguYnguLLguIrguLTguIcgW0VOVElUWV9CXSDguYPguJnguJvguLXguYPguJQiLCJiYWNrX3RyYW5zbGF0aW9uX2VuIjoiW0VOVElUWV9BXSBnb3Qgbm9taW5hdGVkIGZvciB0aGUgW0VOVElUWV9CXSBCb3dsIGluIHdoYXQgeWVhcj8ifSx7InJldmlld19rZXkiOiJSMDYwMiIsImxhbmd1YWdlIjoidGgiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiT24gd2hhdCBkYXRlIGRpZCBbRU5USVRZX0FdIGNlYXNlIHRvIGJlIHRoZSBjb3VudHJ5IG9mIFtFTlRJVFlfQl0/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IltFTlRJVFlfQV0g4Liq4Li04LmJ4LiZ4Liq4Li44LiU4LiB4Liy4Lij4LmA4Lib4LmH4LiZ4Lib4Lij4Liw4LmA4LiX4Lio4LiX4Li14LmIIFtFTlRJVFlfQl0g4LiV4Lix4LmJ4LiH4Lit4Lii4Li54LmI4LmA4Lih4Li34LmI4Lit4Lin4Lix4LiZ4LiX4Li14LmI4LmA4LiX4LmI4Liy4LmD4LiUIiwiYmFja190cmFuc2xhdGlvbl9lbiI6IltFTlRJVFlfQV0gZW5kcyB1cCBiZWluZyB0aGUgY291bnRyeSB3aGVyZSBbRU5USVRZX0JdIGlzIGxvY2F0ZWQgb24gd2hhdCBkYXRlPyJ9LHsicmV2aWV3X2tleSI6IlIwNjc3IiwibGFuZ3VhZ2UiOiJ0aCIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJJbiB3aGF0IHllYXIgZGlkIFtFTlRJVFlfQV0gcmVjZWl2ZSB0aGUgW0VOVElUWV9CXT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiW0VOVElUWV9BXSDguYTguJTguYnguKPguLHguJogW0VOVElUWV9CXSDguYPguJnguJvguLXguYPguJQiLCJiYWNrX3RyYW5zbGF0aW9uX2VuIjoiW0VOVElUWV9BXSBnb3QgW0VOVElUWV9CXSBpbiB3aGF0IHllYXI/In1d"
LANG_NAMES = {"fr":"French", "de":"German", "hi":"Hindi", "th":"Thai"}
MARKER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")
OUT = Path("/kaggle/working/qwen_recheck.json")

SYSTEM = (
    "You are a strict but linguistically informed translation adjudicator for synthetic QA prompts. "
    "For semantic_pass, decide whether the target-language question asks the same thing as the English "
    "source: preserve relation, direction, polarity, scope, tense/time, and answer type. Accept faithful "
    "paraphrases, inflection changes, and contextual synonyms. Do not reject merely because a word has "
    "another dictionary sense; judge the translated question in context. Use the English back-translation "
    "as independent diagnostic evidence, not as a requirement for word-for-word equality. For fluency_pass, "
    "judge only whether the translated question is grammatical and natural in the named target language. "
    "Names/numbers are opaque placeholders. Do not answer the questions or infer their facts. If a material "
    "meaning change or substantial fluency problem is uncertain, return false. Give one concise reason. "
    "Return exactly one decision for the supplied key."
)

def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out

def parse_decision(raw, expected_key):
    text = raw.strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("model response contains no JSON object")
    parsed, end = json.JSONDecoder(object_pairs_hook=unique_object).raw_decode(text[start:])
    if text[start + end:].strip():
        raise ValueError("non-whitespace trailing text after JSON")
    if not isinstance(parsed, dict):
        raise ValueError("top-level response is not an object")
    if set(parsed) == {"decisions"}:
        decisions = parsed["decisions"]
        if not isinstance(decisions, list) or len(decisions) != 1:
            raise ValueError("expected exactly one decision in decisions array")
        decision = decisions[0]
    elif set(parsed) == {expected_key} and isinstance(parsed[expected_key], dict):
        decision = {"key": expected_key, **parsed[expected_key]}
    else:
        decision = parsed
    if (not isinstance(decision, dict)
            or set(decision) != {"key", "semantic_pass", "fluency_pass", "reason"}
            or decision.get("key") != expected_key
            or type(decision.get("semantic_pass")) is not bool
            or type(decision.get("fluency_pass")) is not bool
            or not isinstance(decision.get("reason"), str)
            or not decision["reason"].strip()):
        raise ValueError("single decision failed exact key/schema checks")
    return decision

def main():
    raw_input = base64.b64decode(RAW_INPUT_B64)
    if hashlib.sha256(raw_input).hexdigest() != INPUT_SHA256:
        raise ValueError("embedded input hash mismatch")
    rows = json.loads(raw_input)
    if len(rows) != 31 or len({r.get("review_key") for r in rows}) != 31:
        raise ValueError("expected 31 unique unresolved rows")
    for row in rows:
        if set(row) != {"review_key", "language", "kind", "source_question_en", "translated_question", "back_translation_en"}:
            raise ValueError("unexpected field; only question text is allowed")
        if not (collections.Counter(MARKER.findall(row["source_question_en"]))
                == collections.Counter(MARKER.findall(row["translated_question"]))
                == collections.Counter(MARKER.findall(row["back_translation_en"]))):
            raise ValueError(f"{row['review_key']}: placeholder sequences differ")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; refusing a different-device model review")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
        torch_dtype=torch.float16, quantization_config=quant, device_map="auto")
    model.eval()
    result = {"status":"IN_PROGRESS", "model_id":MODEL_ID, "model_revision":MODEL_REVISION,
        "input_sha256":INPUT_SHA256, "torch":torch.__version__,
        "transformers":transformers.__version__, "accelerate":accelerate.__version__,
        "bitsandbytes":bitsandbytes.__version__, "quantization":"4-bit NF4",
        "gpu":torch.cuda.get_device_name(0), "answers_or_splits_sent":False,
        "human_translation_reviewed":False, "records":[],
        "started_at_utc":datetime.now(timezone.utc).isoformat()}
    for i, row in enumerate(rows, 1):
        payload = {"target_language":LANG_NAMES[row["language"]], "items":[{
            "key":row["review_key"], "source_question_en":row["source_question_en"],
            "translated_question":row["translated_question"],
            "back_translation_en":row["back_translation_en"]}]}
        messages = [{"role":"system", "content":SYSTEM},
                    {"role":"user", "content":json.dumps(payload, ensure_ascii=False)}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tok(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=256, do_sample=False,
                pad_token_id=tok.eos_token_id)
        raw = tok.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            decision = parse_decision(raw, row["review_key"])
        except Exception as exc:
            decision = {"error":type(exc).__name__, "detail":str(exc)}
        record = {"review_key":row["review_key"], "language":row["language"], "kind":row["kind"],
            "source_sha256":sha(row["source_question_en"]),
            "translation_sha256":sha(row["translated_question"]),
            "back_translation_sha256":sha(row["back_translation_en"]),
            "raw_output":raw, "decision":decision}
        result["records"].append(record)
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(OUT)
        print(f"Reviewed {i}/{len(rows)}", flush=True)
        del encoded, generated
        torch.cuda.empty_cache()
    counts = collections.Counter()
    for record in result["records"]:
        d = record["decision"]
        if "error" in d: counts["MALFORMED_BLOCKED"] += 1
        elif d["semantic_pass"] and d["fluency_pass"]: counts["PASS"] += 1
        else: counts["FLAG"] += 1
    result.update({"status":"DIAGNOSTIC_COMPLETE", "counts":dict(counts),
        "completed_at_utc":datetime.now(timezone.utc).isoformat()})
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status":result["status"], "counts":result["counts"]}, indent=2), flush=True)

if __name__ == "__main__":
    main()

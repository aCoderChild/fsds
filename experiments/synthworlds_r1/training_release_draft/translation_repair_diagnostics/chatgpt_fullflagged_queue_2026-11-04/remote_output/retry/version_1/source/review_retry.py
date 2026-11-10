#!/usr/bin/env python3
"""AI-only round-trip and semantic/fluency screen for masked repair questions."""
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

MODE = 'retry_full799_unresolved_v1'
INPUT_SHA256 = '0db7cf44ebbbbc3c35a82a502c9851359068a1555bdc67ba1f0538ee391d38d2'
RAW_INPUT = '[{"kind":"atomic","language":"fr","review_key":"R0011","source_question_en":"On what date did [ENTITY_A] become a relative of [ENTITY_B]?","translated_question":"À quelle date [ENTITY_A] est-il devenu un membre de la famille de [ENTITY_B] ?","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"atomic","language":"fr","review_key":"R0036","source_question_en":"In what year was [ENTITY_A] the architect of [ENTITY_B]?","translated_question":"En quelle année [ENTITY_A] était-il l’architecte de [ENTITY_B] ?","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"atomic","language":"de","review_key":"R0179","source_question_en":"On what date did [ENTITY_A] become the country that [ENTITY_B] is in?","translated_question":"An welchem Datum wurde [ENTITY_A] zu dem Land, in dem sich [ENTITY_B] befindet?","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"atomic","language":"de","review_key":"R0206","source_question_en":"What is the [ENTITY_A] different from?","translated_question":"Wovon unterscheidet sich [ENTITY_A]?","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"atomic","language":"de","review_key":"R0222","source_question_en":"Which entity is [ENTITY_A] different from?","translated_question":"Von welcher Entität unterscheidet sich [ENTITY_A]?","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"atomic","language":"de","review_key":"R0238","source_question_en":"On what date did [ENTITY_A] stop being the capital of [ENTITY_B]?","translated_question":"An welchem Datum hörte [ENTITY_A] auf, die Hauptstadt von [ENTITY_B] zu sein?","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"atomic","language":"hi","review_key":"R0344","source_question_en":"In what year did [ENTITY_A] found [ENTITY_B]?","translated_question":"[ENTITY_A] ने [ENTITY_B] की स्थापना किस वर्ष की थी?","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"atomic","language":"hi","review_key":"R0349","source_question_en":"On what date did [ENTITY_A] become a relative of [ENTITY_B]?","translated_question":"[ENTITY_A] किस तारीख को [ENTITY_B] का रिश्तेदार बना?","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"atomic","language":"hi","review_key":"R0360","source_question_en":"On what date did [ENTITY_A] become the country that [ENTITY_B] is in?","translated_question":"किस तारीख़ को [ENTITY_A] वह देश बना जिसमें [ENTITY_B] स्थित है?","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"atomic","language":"hi","review_key":"R0371","source_question_en":"In what year was [ENTITY_A] nominated for [ENTITY_B]?","translated_question":"[ENTITY_A] को [ENTITY_B] के लिए किस वर्ष नामांकित किया गया था?","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"atomic","language":"hi","review_key":"R0379","source_question_en":"On what date did [ENTITY_A] cease to be the country of [ENTITY_B]?","translated_question":"किस तारीख़ को [ENTITY_A], [ENTITY_B] का देश नहीं रहा?","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"atomic","language":"hi","review_key":"R0407","source_question_en":"In what year did [ENTITY_A] become the spouse of [ENTITY_B]?","translated_question":"[ENTITY_A] किस वर्ष [ENTITY_B] का जीवनसाथी बना?","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"atomic","language":"hi","review_key":"R0417","source_question_en":"On what date did [ENTITY_A] stop being the capital of [ENTITY_B]?","translated_question":"किस तारीख़ को [ENTITY_A], [ENTITY_B] की राजधानी नहीं रहा?","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"atomic","language":"hi","review_key":"R0418","source_question_en":"In what year did [ENTITY_A]\'s term as head of government of [ENTITY_B] end?","translated_question":"[ENTITY_B] के शासनाध्यक्ष के रूप में [ENTITY_A] का कार्यकाल किस वर्ष समाप्त हुआ?","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"atomic","language":"hi","review_key":"R0421","source_question_en":"What immediately follows [ENTITY_A]?","translated_question":"[ENTITY_A] के तुरंत बाद क्या आता है?","reuse_backtranslation_en":"What comes after [ENTITY_A]?","reuse_backtranslation_sha256":"bbef1dd88ecdb1dd6d3ca534cc1ee2eb9aef1bd5b4e8a45a3af03cfa4c08f73d"},{"kind":"atomic","language":"hi","review_key":"R0436","source_question_en":"When did [ENTITY_A] become part of the [ENTITY_B]?","translated_question":"[ENTITY_A] कब [ENTITY_B] का हिस्सा बना?","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"atomic","language":"hi","review_key":"R0442","source_question_en":"In what year did [ENTITY_A] receive the [ENTITY_B]?","translated_question":"[ENTITY_A] को [ENTITY_B] किस वर्ष प्राप्त हुआ था?","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"atomic","language":"hi","review_key":"R0449","source_question_en":"In what year did [ENTITY_A] stop serving as a board member of [ENTITY_B]?","translated_question":"[ENTITY_A] ने किस वर्ष [ENTITY_B] के बोर्ड सदस्य के रूप में कार्य करना बंद किया?","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"atomic","language":"hi","review_key":"R0455","source_question_en":"In what year was [ENTITY_A] a doctoral student of [ENTITY_B]?","translated_question":"[ENTITY_A] किस वर्ष [ENTITY_B] का डॉक्टरेट छात्र था?","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"composition","language":"hi","review_key":"R0481","source_question_en":"Which political office of the state that replaced the country whose capital is [ENTITY_A] is held by its head of state?","translated_question":"जिस देश की राजधानी [ENTITY_A] है, उसका स्थान लेने वाले राज्य के राष्ट्राध्यक्ष के पास उस राज्य का कौन-सा राजनीतिक पद है?","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"composition","language":"hi","review_key":"R0489","source_question_en":"What payment types are accepted by the organizer of the event [ENTITY_A] participated in?","translated_question":"जिस आयोजन में [ENTITY_A] ने भाग लिया था, उसका आयोजक किन भुगतान विधियों को स्वीकार करता है?","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"composition","language":"hi","review_key":"R0493","source_question_en":"Which institution holds the documentation files about the person after whom the city that houses the [ENTITY_A]\'s headquarters is named?","translated_question":"जिस शहर में [ENTITY_A] का मुख्यालय स्थित है, उस शहर का नाम जिस व्यक्ति के नाम पर रखा गया है, उससे संबंधित दस्तावेज़ों की फाइलें किस संस्थान के पास हैं?","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"composition","language":"hi","review_key":"R0512","source_question_en":"What immediately follows [ENTITY_A]?","translated_question":"[ENTITY_A] के तुरंत बाद क्या आता है?","reuse_backtranslation_en":"What comes after [ENTITY_A]?","reuse_backtranslation_sha256":"bbef1dd88ecdb1dd6d3ca534cc1ee2eb9aef1bd5b4e8a45a3af03cfa4c08f73d"},{"kind":"composition","language":"hi","review_key":"R0521","source_question_en":"Where is the entity that borders the place [ENTITY_A] is indigenous to headquartered?","translated_question":"जिस स्थान का [ENTITY_A] मूल निवासी है, उसकी सीमा से लगी इकाई का मुख्यालय कहाँ स्थित है?","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"composition","language":"hi","review_key":"R0537","source_question_en":"Which noble title does [ENTITY_A] hold?","translated_question":"[ENTITY_A] के पास कौन-सी कुलीन उपाधि है?","reuse_backtranslation_en":"What is the title of the [ENTITY_A]?","reuse_backtranslation_sha256":"fc474ec32055b24484f7b5b7a51f5e2906bce8af222e777f99939eeb463cd23d"},{"kind":"atomic","language":"th","review_key":"R0544","source_question_en":"In what year did [ENTITY_A] found [ENTITY_B]?","translated_question":"[ENTITY_A] ก่อตั้ง [ENTITY_B] ในปีใด","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"atomic","language":"th","review_key":"R0553","source_question_en":"When did [ENTITY_A] become part of the [ENTITY_B]?","translated_question":"[ENTITY_A] กลายเป็นส่วนหนึ่งของ [ENTITY_B] เมื่อใด","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"atomic","language":"th","review_key":"R0571","source_question_en":"On what date did [ENTITY_A] become the country that [ENTITY_B] is in?","translated_question":"[ENTITY_A] กลายเป็นประเทศที่ [ENTITY_B] ตั้งอยู่เมื่อวันที่เท่าใด","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"atomic","language":"th","review_key":"R0587","source_question_en":"In what year was [ENTITY_A] nominated for [ENTITY_B]?","translated_question":"[ENTITY_A] ได้รับการเสนอชื่อเข้าชิง [ENTITY_B] ในปีใด","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"atomic","language":"th","review_key":"R0602","source_question_en":"On what date did [ENTITY_A] cease to be the country of [ENTITY_B]?","translated_question":"[ENTITY_A] สิ้นสุดการเป็นประเทศที่ [ENTITY_B] ตั้งอยู่เมื่อวันที่เท่าใด","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"atomic","language":"th","review_key":"R0677","source_question_en":"In what year did [ENTITY_A] receive the [ENTITY_B]?","translated_question":"[ENTITY_A] ได้รับ [ENTITY_B] ในปีใด","reuse_backtranslation_en":null,"reuse_backtranslation_sha256":null},{"kind":"composition","language":"th","review_key":"R0734","source_question_en":"What is the next higher rank above the [ENTITY_A]?","translated_question":"ลำดับชั้นที่สูงกว่า [ENTITY_A] ขึ้นไปหนึ่งระดับคืออะไร","reuse_backtranslation_en":"What is the rank higher than [ENTITY_A] up one level?","reuse_backtranslation_sha256":"b12779b595209cbebde88b3efcd2a0657a502cad2753cc070ac41f28fbfbe5e6"},{"kind":"composition","language":"th","review_key":"R0735","source_question_en":"What is the [ENTITY_A] part of?","translated_question":"[ENTITY_A] เป็นส่วนหนึ่งของอะไร","reuse_backtranslation_en":"[ENTITY_A] is part of what?","reuse_backtranslation_sha256":"e2b94c247fc989261cd97ca70e0caf1285da0a0bb312b20093707cbcacee2fc3"},{"kind":"composition","language":"th","review_key":"R0736","source_question_en":"Which work expands the plot of [ENTITY_A]?","translated_question":"ผลงานใดขยายเนื้อเรื่องของ [ENTITY_A]","reuse_backtranslation_en":"What works expands the content of [ENTITY_A]","reuse_backtranslation_sha256":"fcde1f679ab2008084b0deb37af4859bc6ed14c7cf3bb29bc6b87acc4175a58b"},{"kind":"composition","language":"th","review_key":"R0737","source_question_en":"Which work does [ENTITY_A] reference?","translated_question":"[ENTITY_A] อ้างอิงถึงผลงานใด","reuse_backtranslation_en":"What is the result of [ENTITY_A]?","reuse_backtranslation_sha256":"34806beab001e21c2646497af676d296686ec010830f2cf1450a850542db201d"},{"kind":"composition","language":"th","review_key":"R0738","source_question_en":"Which institution holds the documentation files about the person after whom the city that houses the [ENTITY_A]\'s headquarters is named?","translated_question":"สถาบันใดเก็บรักษาแฟ้มเอกสารเกี่ยวกับบุคคลที่เมืองซึ่งเป็นที่ตั้งสำนักงานใหญ่ของ [ENTITY_A] ได้รับการตั้งชื่อตาม","reuse_backtranslation_en":"Any institution that keeps a file on a person in the city where the [ENTITY_A] headquarters is located is named after him.","reuse_backtranslation_sha256":"28c49f8c05b875ddbb1e7b750a2a19b0805b540a9d41bf987dffb4225753caaa"},{"kind":"composition","language":"th","review_key":"R0739","source_question_en":"Who is [ENTITY_A] named after?","translated_question":"[ENTITY_A] ได้รับการตั้งชื่อตามใคร","reuse_backtranslation_en":"[ENTITY_A] is named after who?","reuse_backtranslation_sha256":"1d0ba7024c39734ce67db4fa3cff0fe88cea6bfcf40f39a065d02a8b966be437"},{"kind":"composition","language":"th","review_key":"R0740","source_question_en":"Where is the mother of the person [ENTITY_A] works for buried?","translated_question":"มารดาของบุคคลที่ [ENTITY_A] ทำงานให้ถูกฝังไว้ที่ไหน","reuse_backtranslation_en":"Where did the mother of the person at [ENTITY_A] work get buried?","reuse_backtranslation_sha256":"08276bf283d007f321deeb8af2685c7b5e65a7c58dde7a39dcfc18ee51778285"},{"kind":"composition","language":"th","review_key":"R0741","source_question_en":"Who does the teacher of [ENTITY_A]\'s father differ from?","translated_question":"ครูของบิดาของ [ENTITY_A] เป็นคนละบุคคลกับใคร","reuse_backtranslation_en":"The teacher of [ENTITY_A]\'s father is a person with whom","reuse_backtranslation_sha256":"804080ea02e05297d4457f19056c0845b2e5f262c84e0772ca515c95adc5333b"},{"kind":"composition","language":"th","review_key":"R0742","source_question_en":"Who is [ENTITY_A] different from?","translated_question":"[ENTITY_A] เป็นคนละบุคคลกับใคร","reuse_backtranslation_en":"[ENTITY_A] is an individual to whom.","reuse_backtranslation_sha256":"c400b08bc3463134b785e5e3ac3c8ffaeb7aa3a8b0987380b590bb1cdddff838"},{"kind":"composition","language":"th","review_key":"R0743","source_question_en":"Who was the doctoral advisor of the rector of [ENTITY_A]\'s employer?","translated_question":"ใครเคยเป็นอาจารย์ที่ปรึกษาระดับปริญญาเอกของอธิการบดีขององค์กรที่จ้างงาน [ENTITY_A]","reuse_backtranslation_en":"Who was the adjunct professor of the administration of the organization that employed [ENTITY_A]","reuse_backtranslation_sha256":"02c207766b2dc974c7d07f95745bd3c98cbcb244c1e11f59d46016436f6db85b"},{"kind":"composition","language":"th","review_key":"R0744","source_question_en":"Who is the rector of [ENTITY_A]?","translated_question":"ใครเป็นอธิการบดีของ [ENTITY_A]","reuse_backtranslation_en":"Who is the governor of [ENTITY_A]?","reuse_backtranslation_sha256":"cf83a2e453c59d45e762b9d15be50258498ec56d0222cb45caf42a263d35d44c"},{"kind":"composition","language":"th","review_key":"R0745","source_question_en":"Who was [ENTITY_A]\'s doctoral advisor?","translated_question":"ใครเคยเป็นอาจารย์ที่ปรึกษาระดับปริญญาเอกของ [ENTITY_A]","reuse_backtranslation_en":"Who was the professor of postdoctoral training at [ENTITY_A]?","reuse_backtranslation_sha256":"d54a5263d9c2ee130b4cba5f6b00a56b509432e733d2084ec4469e05ae069551"},{"kind":"composition","language":"th","review_key":"R0746","source_question_en":"What honorific prefix is used before the name of the unmarried partner of the founder of [ENTITY_A]?","translated_question":"คำยกย่องใดใช้เป็นคำนำหน้าชื่อของคู่รักที่ไม่ได้สมรสของผู้ก่อตั้ง [ENTITY_A]","reuse_backtranslation_en":"What credit is used as the initials of the unmarried couple of the founder of [ENTITY_A]","reuse_backtranslation_sha256":"b6f92e3c71caf0a65bd02c87ac735690d82036ad659442a8e0d67bacf3eacb43"},{"kind":"composition","language":"th","review_key":"R0747","source_question_en":"Who is the unmarried partner of [ENTITY_A]?","translated_question":"ใครเป็นคู่รักที่ไม่ได้สมรสของ [ENTITY_A]","reuse_backtranslation_en":"Who is the unmarried spouse of [ENTITY_A]?","reuse_backtranslation_sha256":"db297442deef8101a98147661876e585e0339ffa6faaf894073a1a2316ae6883"},{"kind":"composition","language":"th","review_key":"R0748","source_question_en":"What honorific prefix is used before [ENTITY_A]\'s name?","translated_question":"คำยกย่องใดใช้เป็นคำนำหน้าชื่อของ [ENTITY_A]","reuse_backtranslation_en":"What is the prefix used for [ENTITY_A]?","reuse_backtranslation_sha256":"35a1710ec563e1bd82bcda83746c7dc51d1c7e15133eac606d4b3ba970782b99"},{"kind":"composition","language":"th","review_key":"R0749","source_question_en":"Who is the unmarried partner of [ENTITY_A]?","translated_question":"ใครเป็นคู่รักที่ไม่ได้สมรสของ [ENTITY_A]","reuse_backtranslation_en":"Who is the unmarried spouse of [ENTITY_A]?","reuse_backtranslation_sha256":"db297442deef8101a98147661876e585e0339ffa6faaf894073a1a2316ae6883"},{"kind":"composition","language":"th","review_key":"R0750","source_question_en":"What is the given name of the director of the work that is part of [ENTITY_A]?","translated_question":"ชื่อตัวของผู้กำกับผลงานที่เป็นส่วนหนึ่งของ [ENTITY_A] คืออะไร","reuse_backtranslation_en":"What is the name of the project manager who is part of [ENTITY_A]?","reuse_backtranslation_sha256":"efd4c82a7ab5603041078a6515e7e6fefedbba797908094d6a75f3ac72eabc88"},{"kind":"composition","language":"th","review_key":"R0751","source_question_en":"Which work is a part of [ENTITY_A]?","translated_question":"ผลงานใดเป็นส่วนหนึ่งของ [ENTITY_A]","reuse_backtranslation_en":"What works is part of [ENTITY_A]","reuse_backtranslation_sha256":"b8a19fd58acb8b424f61dcdc40174806f2b4d3ded4752fd53ea25b022e4d6c45"},{"kind":"composition","language":"th","review_key":"R0752","source_question_en":"What is [ENTITY_A]\'s given name?","translated_question":"ชื่อตัวของ [ENTITY_A] คืออะไร","reuse_backtranslation_en":"What is the name of [ENTITY_A]?","reuse_backtranslation_sha256":"2a764ed05bdbf2224045db1dee0bc3cdccdbacbcdcefdca7c92fc321505a66bb"},{"kind":"composition","language":"th","review_key":"R0753","source_question_en":"Who is the sibling of the person after whom the place that houses the [ENTITY_A] is named?","translated_question":"ใครเป็นพี่น้องของบุคคลที่สถานที่ซึ่งเป็นที่ตั้งของ [ENTITY_A] ได้รับการตั้งชื่อตาม","reuse_backtranslation_en":"Who is the sibling of the person at the place where the [ENTITY_A] is located named after","reuse_backtranslation_sha256":"256e6510962c235b03043f26da12a8bf5550aa47ef04a290bc99d1cb5c5e4a60"},{"kind":"composition","language":"th","review_key":"R0754","source_question_en":"Who is [ENTITY_A] named after?","translated_question":"[ENTITY_A] ได้รับการตั้งชื่อตามใคร","reuse_backtranslation_en":"[ENTITY_A] is named after who?","reuse_backtranslation_sha256":"1d0ba7024c39734ce67db4fa3cff0fe88cea6bfcf40f39a065d02a8b966be437"},{"kind":"composition","language":"th","review_key":"R0755","source_question_en":"Who is [ENTITY_A]\'s sibling?","translated_question":"ใครเป็นพี่น้องของ [ENTITY_A]","reuse_backtranslation_en":"Who are the brothers of [ENTITY_A]?","reuse_backtranslation_sha256":"586301ceab2bfbe20a0caf110792867b34a4fab83a4643097f2c1bab5c9165f7"},{"kind":"composition","language":"th","review_key":"R0756","source_question_en":"Who is the spouse of the godparent of the head of state of [ENTITY_A]?","translated_question":"ใครเป็นคู่สมรสของพ่อแม่อุปถัมภ์ทางศาสนาของประมุขแห่งรัฐของ [ENTITY_A]","reuse_backtranslation_en":"Who is the spouse of the religious adoptive parents of the head of state of [ENTITY_A]","reuse_backtranslation_sha256":"dba020072c9cf45cd51d75504c2ade6e0fc7d937c37803175c1ae1f16f52d946"},{"kind":"composition","language":"th","review_key":"R0757","source_question_en":"Who is [ENTITY_A]\'s godparent?","translated_question":"ใครเป็นพ่อแม่อุปถัมภ์ทางศาสนาของ [ENTITY_A]","reuse_backtranslation_en":"Who are the religious adoptive parents of [ENTITY_A]?","reuse_backtranslation_sha256":"1138dfd2cee096eb6d0d4ded7a089e676f81310ffba1ca75c45071ac63835277"},{"kind":"composition","language":"th","review_key":"R0758","source_question_en":"What country is [ENTITY_A] a citizen of?","translated_question":"[ENTITY_A] เป็นพลเมืองของประเทศใด","reuse_backtranslation_en":"[ENTITY_A] is a citizen of which country?","reuse_backtranslation_sha256":"9fc6ff74ff24581778109918de9a5246d0eb5469daeb2e24dbecd15abc1be45a"},{"kind":"composition","language":"th","review_key":"R0759","source_question_en":"What is the family name of the entity that immediately follows the family to which [ENTITY_A] belongs?","translated_question":"นามสกุลของบุคคลหรือสิ่งที่ตามหลังตระกูลซึ่ง [ENTITY_A] สังกัดอยู่ทันทีคืออะไร","reuse_backtranslation_en":"What is the surname of the person or thing after the family to which [ENTITY_A] immediately belongs?","reuse_backtranslation_sha256":"00a72ad048709c767820a4e16b437af4d831db90cdad4547f92a6a49e014c61f"},{"kind":"composition","language":"th","review_key":"R0760","source_question_en":"What immediately follows [ENTITY_A]?","translated_question":"อะไรตามหลัง [ENTITY_A] ทันที","reuse_backtranslation_en":"What\'s after [ENTITY_A] right now?","reuse_backtranslation_sha256":"c6b723dad1fbf4e3e1ad089fd8c88b9660131e0ba163a30006adc3ec19e1b339"},{"kind":"composition","language":"th","review_key":"R0761","source_question_en":"What is [ENTITY_A]\'s family name?","translated_question":"นามสกุลของ [ENTITY_A] คืออะไร","reuse_backtranslation_en":"What\'s the last name of [ENTITY_A]?","reuse_backtranslation_sha256":"b5e1a216b27b1d115a9d375ac45c2f4bc28fb66559ef77bdf3b899285e394fe4"},{"kind":"composition","language":"th","review_key":"R0762","source_question_en":"Which family does the mother of the main subject of the [ENTITY_A] belong to?","translated_question":"มารดาของบุคคลที่เป็นหัวข้อหลักของ [ENTITY_A] อยู่ในตระกูลใด","reuse_backtranslation_en":"The mother of the person who is the main subject of [ENTITY_A] belongs to which family","reuse_backtranslation_sha256":"f2124be1841ad87bbb920ce71ca2ddcd32e265602977403deb7b816aff03f2dc"},{"kind":"composition","language":"th","review_key":"R0763","source_question_en":"Who is the mother of the father of the person associated with [ENTITY_A]?","translated_question":"ใครเป็นมารดาของบิดาของบุคคลที่เกี่ยวข้องกับ [ENTITY_A]","reuse_backtranslation_en":"Who is the mother of the father of the person involved in [ENTITY_A]?","reuse_backtranslation_sha256":"23c88a699af0e6822311c3f316e31008a86b2508aea94d2022a856f53f4e8364"},{"kind":"composition","language":"th","review_key":"R0764","source_question_en":"What is the given name of the unmarried partner of [ENTITY_A]\'s sibling?","translated_question":"ชื่อตัวของคู่รักที่ไม่ได้สมรสของพี่น้องของ [ENTITY_A] คืออะไร","reuse_backtranslation_en":"What\'s the name of the unmarried couple of [ENTITY_A]\'s siblings?","reuse_backtranslation_sha256":"2a3ab622a990a2dfd550876a787f94a022f2e33d611e82a3db1790b7fd0734d7"},{"kind":"composition","language":"th","review_key":"R0765","source_question_en":"Who is [ENTITY_A]\'s sibling?","translated_question":"ใครเป็นพี่น้องของ [ENTITY_A]","reuse_backtranslation_en":"Who are the brothers of [ENTITY_A]?","reuse_backtranslation_sha256":"586301ceab2bfbe20a0caf110792867b34a4fab83a4643097f2c1bab5c9165f7"},{"kind":"composition","language":"th","review_key":"R0766","source_question_en":"What is [ENTITY_A]\'s given name?","translated_question":"ชื่อตัวของ [ENTITY_A] คืออะไร","reuse_backtranslation_en":"What is the name of [ENTITY_A]?","reuse_backtranslation_sha256":"2a764ed05bdbf2224045db1dee0bc3cdccdbacbcdcefdca7c92fc321505a66bb"},{"kind":"composition","language":"th","review_key":"R0767","source_question_en":"Which institution holds documentation files about the founder of the company owned by [ENTITY_A]?","translated_question":"สถาบันใดเก็บรักษาแฟ้มเอกสารเกี่ยวกับผู้ก่อตั้งบริษัทที่ [ENTITY_A] เป็นเจ้าของ","reuse_backtranslation_en":"Which institution keeps a file about the founder of the company that [ENTITY_A] owns?","reuse_backtranslation_sha256":"96f0ed2bc4eb11d7d835c00ab15dfd4518979489892331da94cdafb343fb513c"},{"kind":"composition","language":"th","review_key":"R0768","source_question_en":"Which entity is owned by [ENTITY_A]?","translated_question":"หน่วยงานหรือสิ่งใดเป็นกรรมสิทธิ์ของ [ENTITY_A]","reuse_backtranslation_en":"The agency or thing is the copyright of [ENTITY_A]","reuse_backtranslation_sha256":"1680c7f681e9cfc91b15eb551f8eae3457e77525d86a2898dfd4a60540b28a4a"},{"kind":"composition","language":"th","review_key":"R0769","source_question_en":"Where is the entity that borders the place [ENTITY_A] is indigenous to headquartered?","translated_question":"สำนักงานใหญ่ของหน่วยงานที่มีพรมแดนติดกับสถานที่ซึ่ง [ENTITY_A] มีถิ่นกำเนิดดั้งเดิมตั้งอยู่ที่ไหน","reuse_backtranslation_en":"The headquarters of the agency is located adjacent to where [ENTITY_A] originally came from.","reuse_backtranslation_sha256":"0090b805514f2fb97097783dd87b63b06d457814c719c75ee639757d9702a0d6"},{"kind":"composition","language":"th","review_key":"R0770","source_question_en":"Where is [ENTITY_A] indigenous to?","translated_question":"[ENTITY_A] มีถิ่นกำเนิดดั้งเดิมอยู่ที่ไหน","reuse_backtranslation_en":"Where did [ENTITY_A] originate from?","reuse_backtranslation_sha256":"922a1208df5d9869286637b209d401b3003e7e7d255a67c3c3345a0aa7f78c1f"},{"kind":"composition","language":"th","review_key":"R0771","source_question_en":"What shares a border with [ENTITY_A]?","translated_question":"อะไรมีพรมแดนติดกับ [ENTITY_A]","reuse_backtranslation_en":"What\'s the boundary with [ENTITY_A]?","reuse_backtranslation_sha256":"1eb726c8859747520a3d1c8fea9af19e6e770418ca579103cdfdd1a45b7632c7"},{"kind":"composition","language":"th","review_key":"R0772","source_question_en":"What is the religion of the child of [ENTITY_A]\'s unmarried partner?","translated_question":"บุตรของคู่รักที่ไม่ได้สมรสของ [ENTITY_A] นับถือศาสนาอะไร","reuse_backtranslation_en":"The son of a couple who did not marry [ENTITY_A] respects what religion","reuse_backtranslation_sha256":"aa54e10096842a5da678b6db4cc2394271d6aa0bc06d9f7d44b330b2791e13ca"},{"kind":"composition","language":"th","review_key":"R0773","source_question_en":"Who is the child of [ENTITY_A]?","translated_question":"ใครเป็นบุตรของ [ENTITY_A]","reuse_backtranslation_en":"Who is the son of [ENTITY_A]?","reuse_backtranslation_sha256":"930d4246c0a59bf8de4fcff017d2c0aa0e307ee9fe265194e3353bc0e104b214"},{"kind":"composition","language":"th","review_key":"R0774","source_question_en":"What country is the significant person associated with [ENTITY_A]\'s field of work a citizen of?","translated_question":"บุคคลสำคัญที่เกี่ยวข้องกับสาขาการทำงานของ [ENTITY_A] เป็นพลเมืองของประเทศใด","reuse_backtranslation_en":"The key persons involved in the work of [ENTITY_A] are citizens of which country?","reuse_backtranslation_sha256":"609fdc62a28a478e337b1ad1e39f5437734e5991301ef70a7b79bd0570a36970"},{"kind":"composition","language":"th","review_key":"R0775","source_question_en":"What country is [ENTITY_A] a citizen of?","translated_question":"[ENTITY_A] เป็นพลเมืองของประเทศใด","reuse_backtranslation_en":"[ENTITY_A] is a citizen of which country?","reuse_backtranslation_sha256":"9fc6ff74ff24581778109918de9a5246d0eb5469daeb2e24dbecd15abc1be45a"},{"kind":"composition","language":"th","review_key":"R0776","source_question_en":"Who represents the creator of the participant in [ENTITY_A]?","translated_question":"ใครเป็นผู้แทนของผู้สร้างสรรค์ผลงานของผู้เข้าร่วม [ENTITY_A]","reuse_backtranslation_en":"Who represented the creator of the works of the participants [ENTITY_A]","reuse_backtranslation_sha256":"377c14f7c8cce4e239c01d5dfb09700cb42dedd32a1a08049694ad0a847ae19c"},{"kind":"composition","language":"th","review_key":"R0777","source_question_en":"Who participated in [ENTITY_A]?","translated_question":"ใครเข้าร่วม [ENTITY_A]","reuse_backtranslation_en":"Who is joining [ENTITY_A]?","reuse_backtranslation_sha256":"8a711c19e52b63b52c940c6a46b00773be3c13cd2e917f936c5d2737f5c89c9c"},{"kind":"composition","language":"th","review_key":"R0778","source_question_en":"Who is the spouse of the person for whom the place where [ENTITY_A] worked was named?","translated_question":"ใครเป็นคู่สมรสของบุคคลที่สถานที่ซึ่ง [ENTITY_A] เคยทำงานได้รับการตั้งชื่อตาม","reuse_backtranslation_en":"Who is the spouse of the person at the place where [ENTITY_A] once worked was named after","reuse_backtranslation_sha256":"230e0e93a3f2987a3419d9d8330ac05b6e400531a8488d9b6d84a1112bd9ee87"},{"kind":"composition","language":"th","review_key":"R0779","source_question_en":"Where did [ENTITY_A] work?","translated_question":"[ENTITY_A] เคยทำงานที่ไหน","reuse_backtranslation_en":"Where did [ENTITY_A] work?","reuse_backtranslation_sha256":"05c74ec2ee39fc5e5af1f3c22ab115e166e675e556a75d3879d62c6f1aa48219"},{"kind":"composition","language":"th","review_key":"R0780","source_question_en":"What language does the stepparent of the founder of [ENTITY_A] speak?","translated_question":"พ่อเลี้ยงหรือแม่เลี้ยงของผู้ก่อตั้ง [ENTITY_A] พูดภาษาอะไร","reuse_backtranslation_en":"What language does the foster father or stepmother of the founder of [ENTITY_A] speak?","reuse_backtranslation_sha256":"03bc8b45cf74ff8d5b894692d473f9bb4034628e3bf62d84def1dbc6666f7256"},{"kind":"composition","language":"th","review_key":"R0781","source_question_en":"Who is [ENTITY_A]\'s stepparent?","translated_question":"ใครเป็นพ่อเลี้ยงหรือแม่เลี้ยงของ [ENTITY_A]","reuse_backtranslation_en":"Who is the foster father or foster mother of [ENTITY_A]?","reuse_backtranslation_sha256":"358de7a22ab83be7e090b4ad8e986630e81525dc362235961b904f2b2ba074bb"},{"kind":"composition","language":"th","review_key":"R0782","source_question_en":"Who is the significant person linked to the mother of the person to whom the [ENTITY_A] is dedicated?","translated_question":"ใครเป็นบุคคลสำคัญที่เกี่ยวข้องกับมารดาของบุคคลที่ [ENTITY_A] ได้รับการอุทิศให้","reuse_backtranslation_en":"Who is the important person associated with the mother of the person to whom [ENTITY_A] has been dedicated?","reuse_backtranslation_sha256":"708ae9d3bd3159c50e2538553b29b16a424988c2d2eaaa9122f7188c687faec3"},{"kind":"composition","language":"th","review_key":"R0783","source_question_en":"To whom is the [ENTITY_A] dedicated?","translated_question":"[ENTITY_A] สร้างขึ้นเพื่ออุทิศให้แก่ใคร","reuse_backtranslation_en":"[ENTITY_A] was created to honor who?","reuse_backtranslation_sha256":"10dd611761e474a28a22b86fe3a352d5726faca9b50e8a03275c5b192038cd84"},{"kind":"composition","language":"th","review_key":"R0784","source_question_en":"Who is a significant person linked to [ENTITY_A]?","translated_question":"ใครเป็นบุคคลสำคัญที่มีความเกี่ยวข้องกับ [ENTITY_A]","reuse_backtranslation_en":"Who is the key person involved in [ENTITY_A]?","reuse_backtranslation_sha256":"d494200296ac5b5c84f4d8df4c96b2c6d194923b788457fb61ab112a898c9cf1"},{"kind":"composition","language":"th","review_key":"R0785","source_question_en":"In which conflict did the person who commissioned the institution that holds [ENTITY_A]\'s collection items participate?","translated_question":"บุคคลที่ว่าจ้างให้สร้างสถาบันซึ่งเก็บรักษาวัตถุในคอลเลกชันของ [ENTITY_A] เข้าร่วมในความขัดแย้งใด","reuse_backtranslation_en":"The person who was hired to build the institute that preserves the objects in the collection of [ENTITY_A] participated in what conflict","reuse_backtranslation_sha256":"e8fef6babd48a67050a57238a9059ed335173c18d4391acc0c57daf0527f5049"},{"kind":"composition","language":"th","review_key":"R0786","source_question_en":"What institution holds the collection items of [ENTITY_A]?","translated_question":"สถาบันใดเก็บรักษาวัตถุในคอลเลกชันของ [ENTITY_A]","reuse_backtranslation_en":"What institution holds the objects in the collection of [ENTITY_A]?","reuse_backtranslation_sha256":"f24951f75cf707e7188d3f529c5485d69fc4e3fe3bdef4a34211c46871017bb4"},{"kind":"composition","language":"th","review_key":"R0787","source_question_en":"Who commissioned [ENTITY_A]?","translated_question":"ใครเป็นผู้ว่าจ้างให้สร้าง [ENTITY_A]","reuse_backtranslation_en":"Who hired the governor to build the [ENTITY_A]?","reuse_backtranslation_sha256":"f878380265738802b926f1887e5e84794474feda417b1150294bdcd92b3dc092"},{"kind":"composition","language":"th","review_key":"R0788","source_question_en":"In which conflict did [ENTITY_A] participate?","translated_question":"[ENTITY_A] เข้าร่วมในความขัดแย้งใด","reuse_backtranslation_en":"[ENTITY_A] participated in what conflict","reuse_backtranslation_sha256":"cf7da3bfcf7acf632c1f00c9be4d17b542474b6aa46256cf4169b45fedd4678c"},{"kind":"composition","language":"th","review_key":"R0789","source_question_en":"Which award did the family of [ENTITY_A]\'s child receive?","translated_question":"ครอบครัวของบุตรของ [ENTITY_A] ได้รับรางวัลใด","reuse_backtranslation_en":"What award did the family of [ENTITY_A]\'s son receive?","reuse_backtranslation_sha256":"c296bcc3dafe0a589fdae1adeaf39b521f32e98d3fd332b3eb9256862a73f647"},{"kind":"composition","language":"th","review_key":"R0790","source_question_en":"Which award did [ENTITY_A] receive?","translated_question":"[ENTITY_A] ได้รับรางวัลใด","reuse_backtranslation_en":"What award did [ENTITY_A] win?","reuse_backtranslation_sha256":"7a7a0f448e8540996fabb643f48c28381f7d0b5f59e97606685f688138d7ed4c"},{"kind":"composition","language":"th","review_key":"R0791","source_question_en":"Who is the father of the creator of the noble title held by [ENTITY_A]?","translated_question":"ใครเป็นบิดาของผู้สร้างบรรดาศักดิ์ที่ [ENTITY_A] ดำรงอยู่","reuse_backtranslation_en":"Who is the father of the creator of the gods that [ENTITY_A] holds?","reuse_backtranslation_sha256":"4aeb261c96d0fe86b6f4f3da2de39c333c162fa6733b5a5d5402006ac0b6b06f"},{"kind":"composition","language":"th","review_key":"R0792","source_question_en":"Which noble title does [ENTITY_A] hold?","translated_question":"[ENTITY_A] ดำรงบรรดาศักดิ์ใด","reuse_backtranslation_en":"[ENTITY_A] holds what sacredness","reuse_backtranslation_sha256":"f298ec34b8a553030496057f4db6f8784542034943826b868d720baee3a6698f"},{"kind":"composition","language":"th","review_key":"R0793","source_question_en":"What is the capital of the country that the administrative region containing [ENTITY_A] has diplomatic relations with?","translated_question":"เมืองหลวงของประเทศที่มีความสัมพันธ์ทางการทูตกับเขตการปกครองซึ่งมี [ENTITY_A] อยู่ภายในคือเมืองใด","reuse_backtranslation_en":"The capital of the country with diplomatic relations with the administrative region where [ENTITY_A] is located is which city","reuse_backtranslation_sha256":"53eef127f2608e513aa9b13b27b02853b4810dda3d7a7bb82df069d478fb986d"},{"kind":"composition","language":"th","review_key":"R0794","source_question_en":"What is the capital of the [ENTITY_A]?","translated_question":"เมืองหลวงของ [ENTITY_A] คือเมืองใด","reuse_backtranslation_en":"What is the capital of [ENTITY_A]?","reuse_backtranslation_sha256":"95f670b44f7930991ae57f0eb42d8c2953c2f00f83d61aa585772ea9a196072f"},{"kind":"composition","language":"th","review_key":"R0795","source_question_en":"Where did the father of the godparent of [ENTITY_A] die?","translated_question":"บิดาของพ่อแม่อุปถัมภ์ทางศาสนาของ [ENTITY_A] เสียชีวิตที่ไหน","reuse_backtranslation_en":"Where did the father of [ENTITY_A]\'s religious adoptive parents die?","reuse_backtranslation_sha256":"db6505350376224a93fff416bc04153aa4e5c1f638bf169549eec7f108f46a4a"},{"kind":"composition","language":"th","review_key":"R0796","source_question_en":"Who is the godparent of [ENTITY_A]?","translated_question":"ใครเป็นพ่อแม่อุปถัมภ์ทางศาสนาของ [ENTITY_A]","reuse_backtranslation_en":"Who are the religious adoptive parents of [ENTITY_A]?","reuse_backtranslation_sha256":"1138dfd2cee096eb6d0d4ded7a089e676f81310ffba1ca75c45071ac63835277"},{"kind":"composition","language":"th","review_key":"R0797","source_question_en":"Where does the curator of the collection that has works by [ENTITY_A] reside?","translated_question":"ภัณฑารักษ์ของคอลเลกชันที่มีผลงานของ [ENTITY_A] พำนักอาศัยอยู่ที่ไหน","reuse_backtranslation_en":"The protected area of the collection with works of [ENTITY_A] where the inhabitants live","reuse_backtranslation_sha256":"a45fe9a1445b318c07b1f0035ed0e2820100b43d9ad9c6cd24f3d54bade65e1d"},{"kind":"composition","language":"th","review_key":"R0798","source_question_en":"Which collection has works by [ENTITY_A]?","translated_question":"คอลเลกชันใดมีผลงานของ [ENTITY_A]","reuse_backtranslation_en":"What collection is the work of [ENTITY_A]?","reuse_backtranslation_sha256":"f77606c85e38468042a601e4e332ab31d3893718696c3abc000cafd99c2dd7e9"},{"kind":"composition","language":"th","review_key":"R0799","source_question_en":"Who is the curator of [ENTITY_A]?","translated_question":"ใครเป็นภัณฑารักษ์ของ [ENTITY_A]","reuse_backtranslation_en":"Who is the protector of [ENTITY_A]?","reuse_backtranslation_sha256":"254512bd6b72e0da140548b035b4b63c4021d53ece361e9ef2255eef5b3f7768"}]'
NLLB_ID = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
QWEN_ID = "Qwen/Qwen2.5-14B-Instruct"
QWEN_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
LANGUAGE_NAMES = {"fr": "French", "de": "German", "hi": "Hindi", "th": "Thai"}
NLLB_SOURCE_CODES = {"fr": "fra_Latn", "de": "deu_Latn", "hi": "hin_Deva", "th": "tha_Thai"}
PLACEHOLDER = re.compile(r"\[(ENTITY|NUMBER)_([A-Z]+)\]")
OUT = Path("/kaggle/working/repair_review.json")


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def save(result):
    temp = OUT.with_suffix(".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(OUT)


def mask_placeholders(text):
    counts = Counter()
    originals = {}
    def replace(match):
        token = f"ZXQ{match.group(1)}{match.group(2)}QXZ"
        counts[token] += 1
        originals[token] = match.group(0)
        return token
    return PLACEHOLDER.sub(replace, text), originals, counts


def restore_placeholders(text, originals, expected):
    found = Counter({token: text.count(token) for token in originals})
    if found != expected:
        raise ValueError(f"backtranslation changed opaque placeholders: expected={dict(expected)}, found={dict(found)}")
    for token, original in originals.items():
        text = text.replace(token, original)
    if any(token in text for token in originals):
        raise ValueError("unrestored opaque marker")
    return text.strip()


def checkpoint(status, rows, records, runtime):
    save({"status": status, "mode": MODE, "input_sha256": INPUT_SHA256,
          "nllb_model_id": NLLB_ID, "nllb_model_revision": NLLB_REVISION,
          "qwen_model_id": QWEN_ID, "qwen_model_revision": QWEN_REVISION,
          "records": records, **runtime})


def make_qwen_schema():
    item = {"type": "object", "properties": {
        "key": {"type": "string"}, "semantic_pass": {"type": "boolean"},
        "fluency_pass": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["key", "semantic_pass", "fluency_pass", "reason"], "additionalProperties": False}
    return {"type": "object", "properties": {
        "decisions": {"type": "array", "items": item, "minItems": 1, "maxItems": 1}},
        "required": ["decisions"], "additionalProperties": False}


def judge_one(row, backtranslation, model, tokenizer, prefix_function):
    system = (
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
    payload = {"target_language": LANGUAGE_NAMES[row["language"]], "items": [{
        "key": row["review_key"], "source_question_en": row["source_question_en"],
        "translated_question": row["translated_question"], "back_translation_en": backtranslation}]}
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=512, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id,
                                   prefix_allowed_tokens_fn=prefix_function)
    raw = tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    match = re.search(r"\{.*\}", raw, re.S)
    parsed = json.loads(match.group(0), object_pairs_hook=unique_object) if match else None
    decisions = parsed.get("decisions") if isinstance(parsed, dict) else None
    if not isinstance(decisions, list) or len(decisions) != 1:
        raise ValueError("Qwen returned invalid decisions coverage")
    decision = decisions[0]
    if (not isinstance(decision, dict) or decision.get("key") != row["review_key"]
            or type(decision.get("semantic_pass")) is not bool
            or type(decision.get("fluency_pass")) is not bool
            or not isinstance(decision.get("reason"), str) or not decision["reason"].strip()):
        raise ValueError("Qwen returned an invalid decision schema or key")
    return raw, decision


def main():
    rows = json.loads(RAW_INPUT)
    if sha(RAW_INPUT) != INPUT_SHA256:
        raise ValueError("embedded input hash mismatch")
    if not rows or len({r.get("review_key") for r in rows}) != len(rows):
        raise ValueError("input has no rows or duplicate review keys")
    for row in rows:
        required = {"review_key", "language", "kind", "source_question_en", "translated_question",
                    "reuse_backtranslation_en", "reuse_backtranslation_sha256"}
        if (set(row) != required or row["language"] not in LANGUAGE_NAMES
                or row["kind"] not in {"atomic", "composition"}
                or not row["source_question_en"].strip() or not row["translated_question"].strip()
                or ((row["reuse_backtranslation_en"] is None) != (row["reuse_backtranslation_sha256"] is None))):
            raise ValueError(f"invalid blinded review input: {row.get('review_key')}")
        if row["reuse_backtranslation_en"] is not None:
            reused = row["reuse_backtranslation_en"]
            markers = Counter(PLACEHOLDER.findall(reused))
            if (not isinstance(reused, str) or not reused.strip()
                    or sha(reused) != row["reuse_backtranslation_sha256"]
                    or markers != Counter(PLACEHOLDER.findall(row["source_question_en"]))
                    or markers != Counter(PLACEHOLDER.findall(row["translated_question"]))):
                raise ValueError(f"invalid reusable backtranslation for {row.get('review_key')}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; refusing a slow or different-device run")

    runtime = {"torch": torch.__version__, "transformers": transformers.__version__,
               "accelerate": accelerate.__version__, "bitsandbytes": bitsandbytes.__version__,
               "quantization": "4-bit NF4", "gpu": torch.cuda.get_device_name(0),
               "human_translation_reviewed": False, "answers_or_splits_sent": False,
               "started_at_utc": datetime.now(timezone.utc).isoformat()}
    records = []
    bt_tokenizer = NllbTokenizerFast.from_pretrained(NLLB_ID, revision=NLLB_REVISION)
    bt_model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_ID, revision=NLLB_REVISION, torch_dtype=torch.float16, device_map="auto")
    bt_model.eval()
    encoded = generated = translated = None
    for index, row in enumerate(rows, 1):
        record = {"review_key": row["review_key"], "language": row["language"], "kind": row["kind"],
                  "source_sha256": sha(row["source_question_en"]),
                  "translation_sha256": sha(row["translated_question"]),
                  "backtranslation_en": "", "backtranslation_error": None, "decision": None, "raw_output": None}
        reused = row["reuse_backtranslation_en"]
        if reused is not None:
            record["backtranslation_en"] = reused
            record["backtranslation_sha256"] = sha(reused)
            records.append(record)
            checkpoint("IN_PROGRESS_BACKTRANSLATION", rows, records, runtime)
            print(f"Reused verified backtranslation {index}/{len(rows)}", flush=True)
            continue
        try:
            protected, originals, expected = mask_placeholders(row["translated_question"])
            bt_tokenizer.src_lang = NLLB_SOURCE_CODES[row["language"]]
            encoded = bt_tokenizer(protected, return_tensors="pt", truncation=True, max_length=256).to(bt_model.device)
            with torch.inference_mode():
                generated = bt_model.generate(**encoded,
                    forced_bos_token_id=bt_tokenizer.convert_tokens_to_ids("eng_Latn"),
                    max_new_tokens=128, do_sample=False)
            translated = bt_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            record["backtranslation_en"] = restore_placeholders(translated, originals, expected)
            if not record["backtranslation_en"]:
                raise ValueError("empty backtranslation")
            record["backtranslation_sha256"] = sha(record["backtranslation_en"])
        except Exception as exc:
            record["backtranslation_error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        checkpoint("IN_PROGRESS_BACKTRANSLATION", rows, records, runtime)
        print(f"Backtranslated {index}/{len(rows)}", flush=True)

    encoded = generated = translated = None
    del bt_model, bt_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, revision=QWEN_REVISION)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_ID, revision=QWEN_REVISION, torch_dtype=torch.float16,
        quantization_config=quantization, device_map="auto")
    qwen_model.eval()
    prefix_function = build_transformers_prefix_allowed_tokens_fn(qwen_tokenizer, JsonSchemaParser(make_qwen_schema()))
    record_by_key = {r["review_key"]: r for r in records}
    for index, row in enumerate(rows, 1):
        record = record_by_key[row["review_key"]]
        if record["backtranslation_error"]:
            record["decision"] = {"error": "backtranslation_failed; semantic decision withheld"}
        else:
            last_error = None
            for attempt in range(2):
                try:
                    raw, decision = judge_one(row, record["backtranslation_en"], qwen_model, qwen_tokenizer, prefix_function)
                    record["raw_output"], record["decision"] = raw, decision
                    break
                except Exception as exc:
                    last_error = exc
            if record["decision"] is None:
                record["decision"] = {"error": f"{type(last_error).__name__}: {last_error}"}
        checkpoint("IN_PROGRESS_QWEN14B_REVIEW", rows, records, runtime)
        print(f"Reviewed {index}/{len(rows)}", flush=True)

    blocked = sum(
        "error" in (r.get("decision") or {}) or r.get("backtranslation_error") is not None
        for r in records
    )
    runtime["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    runtime["reviewed_rows"] = sum(isinstance(r.get("decision"), dict) and "error" not in r["decision"] for r in records)
    runtime["blocked_rows"] = blocked
    runtime["status"] = "AI_ONLY_REPAIR_REVIEW_COMPLETE_DIAGNOSTIC_ONLY" if blocked == 0 else "AI_ONLY_REPAIR_REVIEW_HAS_BLOCKED_ROWS"
    save({"status": runtime["status"], "mode": MODE, "input_sha256": INPUT_SHA256,
          "nllb_model_id": NLLB_ID, "nllb_model_revision": NLLB_REVISION,
          "qwen_model_id": QWEN_ID, "qwen_model_revision": QWEN_REVISION,
          "records": records, **runtime})
    print(json.dumps({"status": runtime["status"], "input_rows": len(rows),
                      "reviewed_rows": runtime["reviewed_rows"], "blocked_rows": blocked,
                      "gpu": runtime["gpu"], "quantization": runtime["quantization"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

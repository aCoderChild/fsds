#!/usr/bin/env python3
"""Create NLLB translation and back-translation drafts for disjoint SFT rows.

This utility never approves translations. Its outputs are evidence for the
separate AI-only review and corpus-build steps.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from draft_sft_translations_nllb import LANG_CODES, MODEL_ID


NUMBER = re.compile(r"\d+(?:[-/.]\d+)*")
NUMERIC_TEMPORAL_ANSWER = re.compile(r"\s*\d+(?:[-/.]\d+)*(?:\s*\([A-Za-z][A-Za-z -]*\))?\s*")
MARKER = re.compile(r"(?:ENTITY|NUMBER)[A-Z]{3}|__?(?:ENTITY|NUMBER)_\d{3}__?|<\s*(?:ENTITY|NUMBER)_\d{3}\s*>")
AI_ONLY_REVIEW_CORRECTIONS = {
    "de": {
        "disjoint:0004-qa-sm:q2": ("Welcher Netzsteckertyp ist in Nytheria der Standard?", "Clarified electrical plug type."),
        "disjoint:0004-qa-sm:multi": ("Welcher Netzsteckertyp ist an dem Ort Standard, der an Floria grenzt?", "Clarified electrical plug type and border relation."),
        "disjoint:0011-qa-sm:q2": ("Welche Sprache ist als eine der Komponenten von Pyrathen aufgeführt?", "Restored the requested answer type, language."),
        "disjoint:0011-qa-sm:multi": ("Welche Sprache ist als Komponente von Pyrathen aufgeführt, das zu Quindra Pandyas Arbeitsgebieten gehört?", "Restored the requested answer type and field-of-work relation."),
        "disjoint:0018-qa-sm:q2": ("Zu welcher Teilstreitkraft gehört Valorian Krestorianus Xylanthius?", "Corrected military branch mistranslated as military frontier."),
        "disjoint:0018-qa-sm:multi": ("Zu welcher Teilstreitkraft gehört das Kind von Caspian Valorian?", "Corrected military branch mistranslated as military frontier."),
        "disjoint:0041-qa-sm:q2": ("Von welcher Sprache ist Eldorathian ein Dialekt?", "Made the dialect-of relation explicit."),
        "disjoint:0041-qa-sm:multi": ("Die zum Schreiben von \"Flames of Destiny\" verwendete Sprache ist ein Dialekt welcher Sprache?", "Made the dialect-of relation explicit."),
        "disjoint:0050-qa-sm:q2": ("Welchem Land oder welcher anderen Macht ist Elian Calyssar treu?", "Corrected literal mistranslation of allegiance."),
        "disjoint:0050-qa-sm:multi": ("Welchem Land oder welcher anderen Macht ist die Person treu, für die Calix Silverwood Taufpate ist?", "Corrected allegiance and godparent relations."),
        "disjoint:0076-qa-sm:multi": ("Was besitzt ein Schöpfer, der in The Ephemeral Showcase vorgestellt wird?", "Clarified featured creator without implying a unique creator."),
        "disjoint:0087-qa-sm:q2": ("An welchem Datum wurde Sapphire Bay offiziell als Gemeinde gegründet?", "Preserved municipal incorporation rather than generic founding."),
        "disjoint:0087-qa-sm:multi": ("An welchem Datum wurde der Wohnort von Chyra Sampson offiziell als Gemeinde gegründet?", "Preserved municipal incorporation rather than generic founding."),
        "disjoint:0124-qa-sm:q2": ("Welche Verwaltungseinheit hat Nethoria als Hauptstadt?", "Corrected reversed capital relation."),
        "disjoint:0124-qa-sm:multi": ("Welche Verwaltungseinheit hat die mit Uldarion partnerstädtisch verbundene Stadt als Hauptstadt?", "Corrected reversed capital relation."),
        "disjoint:0125-qa-sm:q2": ("Welche Netzsteckerart ist in Merithia Standard?", "Corrected plug type mistranslated as electric shock."),
        "disjoint:0125-qa-sm:multi": ("Welche Netzsteckerart ist in dem Land Standard, zu dem der Caelum Archipelago diplomatische Beziehungen unterhält?", "Corrected plug type and preserved diplomatic relation."),
        "disjoint:0126-qa-sm:multi": ("In welchem Jahr endete Brenyth Joans Amtszeit als Architekt des Palasts, der mit den Werken von Avelyn Tostado verbunden ist?", "Replaced the misleading description of a palace as a collection."),
        "disjoint:0140-qa-sm:multi": ("In welchem Jahr endete Tarynth Bragas Amtszeit als Vorsitzende der Organisation, der Nytheris Meister angehört?", "Corrected attachment of the organization-membership clause."),
        "disjoint:0146-qa-sm:q2": ("In welchem Monat und Jahr wurde Braxton Goldenvale Bewohner des Opulent Haven Palace?", "Avoided the military sense of occupying a building."),
        "disjoint:0146-qa-sm:multi": ("In welchem Monat und Jahr wurde Braxton Goldenvale Bewohner des Palasts, der zu Delwyn Olearys bemerkenswerten Werken zählt?", "Avoided the military sense of occupying a building."),
        "disjoint:0175-qa-sm:multi": ("In welchem Jahr endete Adeline Everharts Amtszeit in dem Amt, das Irvin Everhart innehatte?", "Clarified the office-holder relation."),
        "disjoint:0183-qa-sm:q2": ("In welchem Jahr wurde Wyrlyn Lord als aktiv verzeichnet?", "Used the biographical sense of active."),
        "disjoint:0183-qa-sm:multi": ("In welchem Jahr wurde der Inhaber des Amts Directorate of Scientific Archives and Research als aktiv verzeichnet?", "Clarified office holder and biographical sense of active."),
        "disjoint:0194-qa-sm:q2": ("Wer war Doktorand unter der Betreuung von Aelynd Shakir?", "Clarified doctoral supervision."),
        "disjoint:0194-qa-sm:multi": ("Wer war Doktorand unter der Betreuung einer mit Wyrden Hays verbundenen Person?", "Clarified doctoral supervision."),
        "disjoint:0196-qa-sm:q2": ("Wie lautet Zyvra Gerbers Vorname?", "Preserved the first-name request."),
        "disjoint:0196-qa-sm:multi": ("Wie lautet der Vorname eines Doktoranden von Tamsin Rivers?", "Preserved first-name and doctoral-student relations."),
        "disjoint:0197-qa-sm:q2": ("Zu welcher Teilstreitkraft gehört Pyridan Ogando?", "Corrected military branch mistranslated as military frontier."),
        "disjoint:0197-qa-sm:multi": ("Zu welcher Teilstreitkraft gehörte der Taufpate von Sylphina Marigold Everwyn?", "Corrected military branch and preserved godparent relation."),
    },
    "fr": {
        "disjoint:0011-qa-sm:q2": ("Quelle langue figure parmi les composantes de Pyrathen ?", "Restored the requested answer type, language."),
        "disjoint:0011-qa-sm:multi": ("Quelle langue figure parmi les composantes de Pyrathen, qui fait partie des domaines de travail de Quindra Pandya ?", "Clarified the field-of-work relation."),
        "disjoint:0041-qa-sm:q2": ("De quelle langue Eldorathian est-il un dialecte ?", "Made the dialect-of relation explicit."),
        "disjoint:0041-qa-sm:multi": ("La langue utilisée pour écrire \"Flames of Destiny\" est un dialecte de quelle langue ?", "Made the dialect-of relation explicit."),
        "disjoint:0050-qa-sm:q2": ("À quel pays ou à quelle autre puissance Elian Calyssar est-il loyal ?", "Used a direct equivalent for allegiance."),
        "disjoint:0050-qa-sm:multi": ("À quel pays ou à quelle autre puissance la personne dont Calix Silverwood est le parrain est-elle loyale ?", "Clarified allegiance and godparent relations."),
        "disjoint:0076-qa-sm:multi": ("Que possède un créateur présenté dans l’exposition The Ephemeral Showcase ?", "Clarified featured creator without implying a unique creator."),
        "disjoint:0087-qa-sm:q2": ("À quelle date Sapphire Bay a-t-elle été officiellement constituée en municipalité ?", "Preserved municipal incorporation."),
        "disjoint:0087-qa-sm:multi": ("À quelle date le lieu où vit Chyra Sampson a-t-il été officiellement constitué en municipalité ?", "Preserved municipal incorporation."),
        "disjoint:0101-qa-sm:q2": ("Quelle œuvre littéraire est à l’origine de Celestial Rework, qui en est une édition ou une traduction ?", "Clarified the source-work direction."),
        "disjoint:0124-qa-sm:q2": ("Quelle division administrative a Nethoria pour capitale ?", "Corrected reversed capital relation."),
        "disjoint:0124-qa-sm:multi": ("Quelle division administrative a pour capitale la ville jumelée avec Uldarion ?", "Corrected reversed capital relation."),
        "disjoint:0125-qa-sm:q2": ("Quel type de fiche électrique est standard en Merithia ?", "Preserved the electrical plug type."),
        "disjoint:0125-qa-sm:multi": ("Quel type de fiche électrique est standard dans le pays qui entretient des relations diplomatiques avec le Caelum Archipelago ?", "Preserved the electrical plug type and diplomatic relation."),
        "disjoint:0126-qa-sm:multi": ("En quelle année le mandat de Brenyth Joan comme architecte a-t-il pris fin au palais associé aux œuvres d’Avelyn Tostado ?", "Replaced the misleading description of a palace as a collection."),
        "disjoint:0140-qa-sm:multi": ("En quelle année le mandat de Tarynth Braga à la présidence de l’organisation dont Nytheris Meister est membre a-t-il pris fin ?", "Clarified the organization-membership clause."),
        "disjoint:0146-qa-sm:q2": ("Quel mois et quelle année marquent le début de l’occupation d’Opulent Haven Palace par Braxton Goldenvale ?", "Avoided the implication that the event was a military occupation."),
        "disjoint:0146-qa-sm:multi": ("Quel mois et quelle année marquent le début de l’occupation du palais répertorié parmi les œuvres remarquables de Delwyn Oleary par Braxton Goldenvale ?", "Avoided the implication that the event was a military occupation."),
        "disjoint:0175-qa-sm:q2": ("En quelle année le mandat d’Adeline Everhart en tant que Duke Everhart of Eldoria a-t-il pris fin ?", "Removed an invalid French contraction before the protected title."),
        "disjoint:0183-qa-sm:q2": ("En quelle année Wyrlyn Lord a-t-il été recensé comme actif ?", "Used the biographical sense of active."),
        "disjoint:0183-qa-sm:multi": ("En quelle année la personne qui occupait le poste de Directorate of Scientific Archives and Research a-t-elle été recensée comme active ?", "Clarified office holder and biographical sense of active."),
        "disjoint:0194-qa-sm:q2": ("Qui était doctorant sous la direction d’Aelynd Shakir ?", "Clarified doctoral supervision."),
        "disjoint:0194-qa-sm:multi": ("Qui était doctorant sous la direction d’une personne associée à Wyrden Hays ?", "Clarified doctoral supervision."),
        "disjoint:0196-qa-sm:q2": ("Quel est le prénom de Zyvra Gerber ?", "Preserved the first-name request."),
        "disjoint:0196-qa-sm:multi": ("Quel est le prénom d’un doctorant de Tamsin Rivers ?", "Preserved first-name and doctoral-student relations."),
        "disjoint:0197-qa-sm:multi": ("À quelle branche militaire appartenait la personne qui est le parrain ou la marraine de Sylphina Marigold Everwyn ?", "Preserved the gender-neutral godparent relation."),
    },
    "hi": {
        "disjoint:0011-qa-sm:q2": ("Pyrathen के घटकों में किस भाषा को शामिल किया गया है?", "Restored the missing requested answer type, language."),
        "disjoint:0011-qa-sm:multi": ("Quindra Pandya के कार्यक्षेत्रों में शामिल Pyrathen के घटकों में कौन-सी भाषा सूचीबद्ध है?", "Restored the requested answer type and field-of-work relation."),
        "disjoint:0041-qa-sm:q2": ("Eldorathian किस भाषा की बोली है?", "Corrected the dialect relation."),
        "disjoint:0041-qa-sm:multi": ("\"Flames of Destiny\" लिखने के लिए इस्तेमाल की गई भाषा किस भाषा की बोली है?", "Corrected the dialect relation."),
        "disjoint:0050-qa-sm:q2": ("Elian Calyssar किस देश या अन्य शक्ति के प्रति वफादार है?", "Corrected literal mistranslation of allegiance."),
        "disjoint:0050-qa-sm:multi": ("वह व्यक्ति, जिसका गॉडपेरेंट Calix Silverwood है, किस देश या अन्य शक्ति के प्रति वफादार है?", "Corrected allegiance and godparent relations."),
        "disjoint:0076-qa-sm:multi": ("The Ephemeral Showcase में प्रदर्शित किसी निर्माता के स्वामित्व में क्या है?", "Clarified featured creator without implying a unique creator."),
        "disjoint:0087-qa-sm:q2": ("Sapphire Bay को नगरपालिका के रूप में औपचारिक रूप से किस तारीख को शामिल किया गया?", "Preserved municipal incorporation."),
        "disjoint:0087-qa-sm:multi": ("Chyra Sampson के निवास-स्थान को नगरपालिका के रूप में औपचारिक रूप से किस तारीख को शामिल किया गया?", "Preserved municipal incorporation."),
        "disjoint:0101-qa-sm:q2": ("Celestial Rework जिस साहित्यिक कृति का संस्करण या अनुवाद है, वह कौन-सी कृति है?", "Clarified the source-work direction."),
        "disjoint:0122-qa-sm:multi": ("Zephyron से सीमा साझा करने वाले स्थान ने किस चीज़ की जगह ली?", "Corrected replacement mistranslated as representation."),
        "disjoint:0124-qa-sm:q2": ("किस प्रशासनिक क्षेत्र की राजधानी Nethoria है?", "Corrected reversed capital relation."),
        "disjoint:0124-qa-sm:multi": ("Uldarion के साथ जुड़वां शहर किस प्रशासनिक क्षेत्र की राजधानी है?", "Corrected reversed capital relation."),
        "disjoint:0125-qa-sm:q2": ("Merithia में बिजली के प्लग का मानक प्रकार क्या है?", "Preserved electrical plug type."),
        "disjoint:0125-qa-sm:multi": ("Caelum Archipelago के साथ राजनयिक संबंध रखने वाले देश में बिजली के प्लग का मानक प्रकार क्या है?", "Preserved electrical plug type and diplomatic relation."),
        "disjoint:0126-qa-sm:multi": ("Avelyn Tostado की कृतियों से जुड़े महल में Brenyth Joan का वास्तुकार के रूप में कार्यकाल किस वर्ष समाप्त हुआ?", "Replaced the misleading description of a palace as a collection."),
        "disjoint:0140-qa-sm:multi": ("Nytheris Meister जिस संगठन का सदस्य है, उसके अध्यक्ष के रूप में Tarynth Braga का कार्यकाल किस वर्ष समाप्त हुआ?", "Clarified the organization-membership clause."),
        "disjoint:0146-qa-sm:q2": ("Braxton Goldenvale Opulent Haven Palace का निवासी किस महीने और वर्ष में बना?", "Avoided the military sense of occupying a building."),
        "disjoint:0146-qa-sm:multi": ("Delwyn Oleary की उल्लेखनीय कृतियों में शामिल महल का निवासी Braxton Goldenvale किस महीने और वर्ष में बना?", "Avoided the military sense of occupying a building."),
        "disjoint:0175-qa-sm:multi": ("Irvin Everhart द्वारा संभाले गए पद पर Adeline Everhart का कार्यकाल किस वर्ष समाप्त हुआ?", "Clarified the office-holder relation."),
        "disjoint:0183-qa-sm:q2": ("Wyrlyn Lord को किस वर्ष सक्रिय दर्ज किया गया था?", "Corrected active mistranslated as effective."),
        "disjoint:0183-qa-sm:multi": ("Directorate of Scientific Archives and Research के पदधारक को किस वर्ष सक्रिय दर्ज किया गया था?", "Clarified office holder and biographical sense of active."),
        "disjoint:0196-qa-sm:q2": ("Zyvra Gerber का पहला नाम क्या है?", "Preserved the first-name request."),
        "disjoint:0196-qa-sm:multi": ("Tamsin Rivers के किसी डॉक्टरेट छात्र का पहला नाम क्या है?", "Preserved first-name and doctoral-student relations."),
        "disjoint:0197-qa-sm:multi": ("Sylphina Marigold Everwyn के गॉडपेरेंट किस सैन्य शाखा से संबंधित थे?", "Corrected godparent mistranslated as father or stepfather."),
    },
    "th": {
        "disjoint:0004-qa-sm:q2": ("ปลั๊กไฟมาตรฐานที่ใช้ใน Nytheria เป็นชนิดใด?", "Corrected plug type mistranslated as electrical wiring."),
        "disjoint:0004-qa-sm:multi": ("ปลั๊กไฟมาตรฐานที่ใช้ในสถานที่ซึ่งมีพรมแดนติดกับ Floria เป็นชนิดใด?", "Corrected plug type and preserved the border relation."),
        "disjoint:0011-qa-sm:q2": ("ภาษาใดถูกระบุว่าเป็นหนึ่งในองค์ประกอบของ Pyrathen?", "Removed unintended Vietnamese code-switching."),
        "disjoint:0011-qa-sm:multi": ("ภาษาใดถูกระบุว่าเป็นองค์ประกอบของ Pyrathen ซึ่งเป็นหนึ่งในสาขางานของ Quindra Pandya?", "Removed unintended Vietnamese code-switching and clarified the relation."),
        "disjoint:0018-qa-sm:q2": ("Valorian Krestorianus Xylanthius สังกัดเหล่าทัพใด?", "Restored a complete question asking for a military branch."),
        "disjoint:0018-qa-sm:multi": ("บุตรของ Caspian Valorian สังกัดเหล่าทัพใด?", "Restored a complete question asking for a military branch."),
        "disjoint:0028-qa-sm:q2": ("Tarynth Braga เริ่มศึกษาระดับปริญญาโทในปีใด?", "Corrected master's degree mistranslated as law school."),
        "disjoint:0028-qa-sm:multi": ("ประธานของ Myralun Luminara Academy Press เริ่มศึกษาระดับปริญญาโทในปีใด?", "Corrected master's degree mistranslated as law school."),
        "disjoint:0039-qa-sm:q2": ("Pendoria ตั้งอยู่ในเขตการปกครองใด?", "Uses the general administrative-region term."),
        "disjoint:0039-qa-sm:multi": ("สถานที่ซึ่ง Jorith เป็นชนพื้นเมืองตั้งอยู่ในเขตการปกครองใด?", "Uses the general administrative-region term."),
        "disjoint:0041-qa-sm:q2": ("Eldorathian เป็นภาษาถิ่นของภาษาใด?", "Corrected the dialect relation."),
        "disjoint:0041-qa-sm:multi": ("ภาษาที่ใช้เขียน \"Flames of Destiny\" เป็นภาษาถิ่นของภาษาใด?", "Corrected the dialect relation."),
        "disjoint:0050-qa-sm:q2": ("Elian Calyssar สวามิภักดิ์ต่อประเทศหรืออำนาจใด?", "Corrected literal mistranslation of allegiance."),
        "disjoint:0050-qa-sm:multi": ("บุคคลที่มี Calix Silverwood เป็นพ่อแม่อุปถัมภ์สวามิภักดิ์ต่อประเทศหรืออำนาจใด?", "Corrected allegiance and godparent relations."),
        "disjoint:0076-qa-sm:multi": ("ผู้สร้างคนหนึ่งที่ได้รับการนำเสนอใน The Ephemeral Showcase เป็นเจ้าของสิ่งใด?", "Clarified featured creator without implying a unique creator."),
        "disjoint:0087-qa-sm:q2": ("Sapphire Bay ได้รับการจัดตั้งเป็นเทศบาลอย่างเป็นทางการเมื่อวันที่เท่าใด?", "Preserved municipal incorporation."),
        "disjoint:0087-qa-sm:multi": ("สถานที่ที่ Chyra Sampson อาศัยอยู่ได้รับการจัดตั้งเป็นเทศบาลอย่างเป็นทางการเมื่อวันที่เท่าใด?", "Preserved municipal incorporation."),
        "disjoint:0101-qa-sm:q2": ("Celestial Rework เป็นฉบับหรือคำแปลของงานวรรณกรรมเรื่องใด?", "Clarified the source-work direction."),
        "disjoint:0122-qa-sm:multi": ("สถานที่ซึ่งมีพรมแดนติดกับ Zephyron เข้าแทนที่สิ่งใด?", "Corrected replacement mistranslated as representation."),
        "disjoint:0124-qa-sm:q2": ("เขตการปกครองใดมี Nethoria เป็นเมืองหลวง?", "Corrected reversed capital relation."),
        "disjoint:0124-qa-sm:multi": ("เมืองที่เป็นเมืองคู่แฝดกับ Uldarion เป็นเมืองหลวงของเขตการปกครองใด?", "Corrected reversed capital relation."),
        "disjoint:0125-qa-sm:q2": ("ปลั๊กไฟมาตรฐานที่ใช้ใน Merithia เป็นชนิดใด?", "Corrected plug type mistranslated as power line."),
        "disjoint:0125-qa-sm:multi": ("ปลั๊กไฟมาตรฐานที่ใช้ในประเทศซึ่งมีความสัมพันธ์ทางการทูตกับ Caelum Archipelago เป็นชนิดใด?", "Corrected plug type and preserved diplomatic relation."),
        "disjoint:0126-qa-sm:multi": ("วังที่เกี่ยวข้องกับผลงานของ Avelyn Tostado เป็นผลงานสถาปัตยกรรมของ Brenyth Joan ซึ่งสิ้นสุดลงในปีใด?", "Replaced the misleading description of a palace as a collection."),
        "disjoint:0140-qa-sm:multi": ("วาระของ Tarynth Braga ในฐานะประธานองค์กรที่ Nytheris Meister เป็นสมาชิกสิ้นสุดลงในปีใด?", "Clarified the organization-membership clause."),
        "disjoint:0146-qa-sm:q2": ("Braxton Goldenvale เริ่มพำนักอยู่ใน Opulent Haven Palace ในเดือนและปีใด?", "Avoided the military sense of occupying a building."),
        "disjoint:0146-qa-sm:multi": ("Braxton Goldenvale เริ่มพำนักอยู่ในวังที่อยู่ในผลงานสำคัญของ Delwyn Oleary ในเดือนและปีใด?", "Avoided the military sense of occupying a building."),
        "disjoint:0162-qa-sm:multi": ("อันดับที่สูงขึ้นไปถัดจากรางวัลที่ Elwood Kylor Zenth ได้รับคืออันดับใด?", "Preserved the higher-rank relation."),
        "disjoint:0175-qa-sm:multi": ("วาระของ Adeline Everhart ในตำแหน่งที่ Irvin Everhart เคยดำรงอยู่สิ้นสุดลงในปีใด?", "Clarified the office-holder relation."),
        "disjoint:0183-qa-sm:q2": ("Wyrlyn Lord มีบทบาทอยู่ในปีใด?", "Corrected active mistranslated as effective."),
        "disjoint:0183-qa-sm:multi": ("ผู้ดำรงตำแหน่ง Directorate of Scientific Archives and Research มีบทบาทอยู่ในปีใด?", "Clarified office holder and biographical sense of active."),
        "disjoint:0196-qa-sm:q2": ("ชื่อจริงของ Zyvra Gerber คืออะไร?", "Preserved the first-name request."),
        "disjoint:0196-qa-sm:multi": ("ชื่อจริงของนักศึกษาปริญญาเอกคนหนึ่งของ Tamsin Rivers คืออะไร?", "Preserved first-name and doctoral-student relations."),
        "disjoint:0197-qa-sm:multi": ("พ่อแม่อุปถัมภ์ในพิธีศีลล้างบาปของ Sylphina Marigold Everwyn สังกัดเหล่าทัพใด?", "Corrected godparent mistranslated as father or stepfather."),
    },
}


def marker_code(index):
    """Return a stable three-letter suffix (AAA, AAB, ..., ZZZ)."""
    if index >= 26 ** 3:
        raise ValueError("Too many protected spans in one field")
    chars = []
    for _ in range(3):
        chars.append(chr(ord("A") + index % 26))
        index //= 26
    return "".join(reversed(chars))


LOCAL_DIGIT_MAP = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹०१२३४५६७८९௦௧௨௩௪௫௬௭௮௯๐๑๒๓๔๕๖๗๘๙",
    "".join("0123456789" for _ in range(5)),
)


def protect_candidate(text, entity_names, scheme="words", protect_numbers=True):
    mapping, counts = {}, {}
    def make_marker(kind):
        index = len(mapping)
        if scheme == "words":
            return f"{kind}{marker_code(index)}"
        if scheme == "underscore":
            return f"__{kind}_{index:03d}__"
        if scheme == "angle":
            return f"<{kind}_{index:03d}>"
        raise ValueError(f"Unknown protected marker scheme: {scheme}")

    def hide_number(match):
        marker = make_marker("NUMBER")
        mapping[marker], counts[marker] = match.group(0), 1
        return marker

    if protect_numbers:
        text = NUMBER.sub(hide_number, text)
    values = sorted({value for value in entity_names if isinstance(value, str) and value}, key=len, reverse=True)
    for value in values:
        marker = make_marker("ENTITY")
        text, count = re.subn(re.escape(value), marker, text, flags=re.IGNORECASE)
        if count:
            mapping[marker], counts[marker] = value, count
    return text, mapping, counts


def restore_candidate(text, mapping, counts, target_code=None):
    normalizations = []
    if target_code == "fra_Latn":
        # NLLB cannot infer the gender of protected names while translating and
        # sometimes inserts French elisions before them (for example, "l'<NAME>").
        # Remove only those contractions; preserve the protected proper name.
        for marker in mapping:
            clean = re.sub(rf"\bde\s+l['’]\s*{re.escape(marker)}\b", f"de {marker}", text, flags=re.IGNORECASE)
            clean = re.sub(rf"\bd['’]\s*{re.escape(marker)}\b", f"de {marker}", clean, flags=re.IGNORECASE)
            clean = re.sub(rf"\bl['’]\s*{re.escape(marker)}\b", marker, clean, flags=re.IGNORECASE)
            if clean != text:
                normalizations.append("FR_PROPER_NAME_ARTICLE_NORMALIZATION")
            text = clean
    for marker, expected_count in counts.items():
        actual_count = text.count(marker)
        if actual_count != expected_count:
            raise ValueError(f"Protected marker {marker} count changed: expected {expected_count}, got {actual_count}")
        text = text.replace(marker, mapping[marker])
    if MARKER.search(text):
        raise ValueError("A mutated or unresolved protected marker remains")
    return text.strip(), sorted(set(normalizations))


def translate_candidate_batch(model, tokenizer, torch, texts, target_code, device, batch_size, source_code="eng_Latn"):
    tokenizer.src_lang = source_code
    target_id = tokenizer.convert_tokens_to_ids(target_code)
    translated = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=256)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            out = model.generate(**batch, forced_bos_token_id=target_id, max_new_tokens=256, num_beams=4)
        translated.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return translated


def translate_protected_field(model, tokenizer, torch, text, entity_names, target_code, device, batch_size):
    """Translate a protected field, retrying only with alternate exact-copy markers."""
    last_error = None
    number_candidates = NUMBER.findall(text)
    attempts = ((scheme, True) for scheme in ("words", "underscore", "angle"))
    attempts = list(attempts) + [(scheme, False) for scheme in ("words", "underscore", "angle")]
    for scheme, protect_numbers in attempts:
        protected, mapping, counts = protect_candidate(
            text, entity_names, scheme=scheme, protect_numbers=protect_numbers
        )
        try:
            output = translate_candidate_batch(
                model, tokenizer, torch, [protected], target_code, device, batch_size
            )[0]
            output, normalizations = restore_candidate(output, mapping, counts, target_code)
            output = output.translate(LOCAL_DIGIT_MAP)
            if any(output.count(number.translate(LOCAL_DIGIT_MAP)) < text.count(number) for number in number_candidates):
                raise ValueError("A source number was changed or omitted")
            return output, scheme + ("" if protect_numbers else "_without_number_markers"), normalizations
        except ValueError as exc:
            last_error = exc
    raise ValueError(f"All protected-marker schemes failed: {last_error}")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def translate(root, batch_size=8, languages=None):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    root = Path(root)
    source_path = root / "sft_training/disjoint_composition_candidates/english_candidate.jsonl"
    output_root = root / "sft_training/disjoint_composition_candidates/translations"
    back_root = root / "sft_training/disjoint_composition_candidates/backtranslations"
    report_path = output_root / "report.json"
    selected_languages = set(languages or LANG_CODES)
    if not selected_languages or selected_languages - set(LANG_CODES):
        raise ValueError(f"Unsupported or empty language selection: {sorted(selected_languages)}")
    source = read_jsonl(source_path)
    if not source or len({row["sft_id"] for row in source}) != len(source):
        raise ValueError("English candidate rows must be non-empty and uniquely keyed")
    if any(row.get("split") != "train_only" or row.get("do_not_train") is not True for row in source):
        raise ValueError("Only quarantined train_only candidates may enter this translator")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading local NLLB translation model on {device}; source rows={len(source)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang="eng_Latn", local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, dtype=dtype, local_files_only=True).to(device).eval()
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    previous_languages = {}
    if report_path.is_file():
        previous_report = json.loads(report_path.read_text(encoding="utf-8"))
        if previous_report.get("source_sha256") == source_sha256 and previous_report.get("translation_model") == MODEL_ID:
            previous_languages = previous_report.get("languages", {})
        elif selected_languages != set(LANG_CODES):
            raise ValueError("Partial translation reruns require a previous report for the exact same English candidate checksum")
    report = {
        "status": "TRANSLATION_DRAFTS_AND_BACKTRANSLATIONS_PENDING_AI_REVIEW",
        "translation_model": MODEL_ID,
        "backtranslation_model": MODEL_ID,
        "backtranslation_is_independent_model": False,
        "device": device,
        "batch_size": batch_size,
        "source_rows": len(source),
        "source_sha256": source_sha256,
        "approval_claimed": False,
        "human_review_performed": False,
        "numbered_translation_policy": "Protect entity names with exact-copy markers; if NLLB cannot preserve a numeric marker, retry without numeric markers and require every source number to survive digit normalization.",
        "canonical_answer_policy": "Answers that exactly equal a protected entity name, or consist only of a numeric/date label with an optional type annotation, remain in canonical source spelling; other answer text is translated.",
        "backtranslation_policy": "Questions and translated natural-language answers are back-translated with the same NLLB-200 model. Canonical named-entity and numeric/date labels are copied into the audit record and are not counted as back-translations.",
        "languages": {},
    }
    report["languages"].update({lang: item for lang, item in previous_languages.items() if lang not in selected_languages})
    report["ai_review_corrections"] = []

    for language, target_code in LANG_CODES.items():
        if language not in selected_languages:
            continue
        print(f"Translating and back-translating {language} candidates", flush=True)
        forward_inputs, protected = [], []
        for row in source:
            question, qmap, qcounts = protect_candidate(row["question_en"], row["protected_entity_names"])
            answer, amap, acounts = protect_candidate(row["answer_en"], row["protected_entity_names"])
            answer_is_canonical_numeric = bool(NUMERIC_TEMPORAL_ANSWER.fullmatch(row["answer_en"]))
            answer_is_canonical_entity = not answer_is_canonical_numeric and any(
                row["answer_en"].strip().casefold() == name.strip().casefold()
                for name in row["protected_entity_names"]
            )
            forward_inputs.extend((question, answer))
            protected.append((qmap, qcounts, amap, acounts, answer_is_canonical_entity, answer_is_canonical_numeric))

        forward = translate_candidate_batch(model, tokenizer, torch, forward_inputs, target_code, device, batch_size)
        translated, repair_count, marker_schemes, normalization_counts = [], 0, Counter(), Counter()
        ai_review_correction_rows = 0
        canonical_entity_answers, canonical_numeric_answers = 0, 0
        for index, row in enumerate(source):
            qmap, qcounts, amap, acounts, answer_is_canonical_entity, answer_is_canonical_numeric = protected[index]
            try:
                try:
                    question, q_normalizations = restore_candidate(forward[2 * index], qmap, qcounts, target_code)
                    q_scheme = "words"
                except ValueError:
                    question, q_scheme, q_normalizations = translate_protected_field(
                        model, tokenizer, torch, row["question_en"], row["protected_entity_names"],
                        target_code, device, batch_size,
                    )
                    repair_count += 1
                if answer_is_canonical_entity or answer_is_canonical_numeric:
                    # Keep proper-name and numeric/date labels stable across languages.
                    answer = row["answer_en"]
                    a_scheme = "canonical_label_passthrough"
                    a_normalizations = []
                    canonical_entity_answers += int(answer_is_canonical_entity)
                    canonical_numeric_answers += int(answer_is_canonical_numeric and not answer_is_canonical_entity)
                else:
                    try:
                        answer, a_normalizations = restore_candidate(forward[2 * index + 1], amap, acounts, target_code)
                        a_scheme = "words"
                    except ValueError:
                        answer, a_scheme, a_normalizations = translate_protected_field(
                            model, tokenizer, torch, row["answer_en"], row["protected_entity_names"],
                            target_code, device, batch_size,
                        )
                        repair_count += 1
            except ValueError as exc:
                raise ValueError(f"Unsafe protected-text translation for {language}/{row['sft_id']}: {exc}") from exc
            row_corrections = []
            correction = AI_ONLY_REVIEW_CORRECTIONS.get(language, {}).get(row["sft_id"])
            if correction:
                question, reason = correction
                row_corrections.append("AI_ONLY_SEMANTIC_CORRECTION")
                report["ai_review_corrections"].append({
                    "language": language,
                    "sft_id": row["sft_id"],
                    "source_question_en": row["question_en"],
                    "question_translation": question,
                    "reason": reason,
                })
                ai_review_correction_rows += 1
            if not question or not answer:
                raise ValueError(f"Empty translated field for {language}/{row['sft_id']}")
            translated.append({
                **row,
                "language": language,
                "question": question,
                "answer_translation": answer,
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                "translation_model": MODEL_ID,
                "translation_status": "MACHINE_DRAFT_PENDING_AI_REVIEW",
                "review_status": "PENDING",
                "review_mode": "AI_ONLY_PENDING",
                "human_review_status": "NOT_PERFORMED",
                "protected_text_repairs": {"question": [], "answer": []},
                "protected_marker_schemes": {"question": q_scheme, "answer": a_scheme},
                "target_language_normalizations": {"question": q_normalizations, "answer": a_normalizations},
                "ai_review_corrections": row_corrections,
                "canonical_entity_answer_passthrough": answer_is_canonical_entity,
                "canonical_numeric_answer_passthrough": answer_is_canonical_numeric and not answer_is_canonical_entity,
                "do_not_train": True,
            })
            marker_schemes[q_scheme] += 1
            marker_schemes[a_scheme] += 1
            normalization_counts.update(q_normalizations)
            normalization_counts.update(a_normalizations)
        write_jsonl(output_root / f"{language}.jsonl", translated)

        back_inputs, question_back_indices, answer_back_indices = [], [], []
        for row in translated:
            question_back_indices.append(len(back_inputs))
            back_inputs.append(row["question"])
            if row["protected_marker_schemes"]["answer"] == "canonical_label_passthrough":
                answer_back_indices.append(None)
            else:
                answer_back_indices.append(len(back_inputs))
                back_inputs.append(row["answer_translation"])
        back = translate_candidate_batch(model, tokenizer, torch, back_inputs, "eng_Latn", device, batch_size, source_code=target_code)
        back_rows = []
        for index, row in enumerate(translated):
            answer_back_index = answer_back_indices[index]
            back_rows.append({
                "sft_id": row["sft_id"],
                "source_qa_id": row["source_qa_id"],
                "language": language,
                "question_en": row["question_en"],
                "question_translation": row["question"],
                "question_backtranslation_en": back[question_back_indices[index]],
                "question_backtranslation_method": "NLLB_200_SAME_MODEL_FAMILY",
                "answer_en": row["answer_en"],
                "answer_translation": row["answer_translation"],
                "answer_backtranslation_en": row["answer_en"] if answer_back_index is None else back[answer_back_index],
                "answer_backtranslation_method": "CANONICAL_LABEL_COPY" if answer_back_index is None else "NLLB_200_SAME_MODEL_FAMILY",
                "translation_model": MODEL_ID,
                "backtranslation_model": MODEL_ID if answer_back_index is not None else None,
                "review_status": "PENDING",
            })
        write_jsonl(back_root / f"{language}.jsonl", back_rows)
        print(f"Completed {language}: {len(translated)} translation rows; {len(back_inputs)} model backtranslations", flush=True)
        report["languages"][language] = {
            "translation_rows": len(translated),
            "backtranslated_fields": len(back_inputs),
            "canonical_label_fields_copied": sum(value is None for value in answer_back_indices),
            "protected_text_repair_rows": repair_count,
            "protected_marker_scheme_field_counts": dict(sorted(marker_schemes.items())),
            "canonical_entity_answer_passthrough_rows": canonical_entity_answers,
            "canonical_numeric_answer_passthrough_rows": canonical_numeric_answers,
            "target_language_normalization_counts": dict(sorted(normalization_counts.items())),
            "ai_review_correction_rows": ai_review_correction_rows,
            "translation_status": "MACHINE_DRAFT_PENDING_AI_REVIEW",
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--languages", nargs="+", choices=tuple(LANG_CODES), default=tuple(LANG_CODES))
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    print(json.dumps(translate(args.root, args.batch_size, args.languages), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

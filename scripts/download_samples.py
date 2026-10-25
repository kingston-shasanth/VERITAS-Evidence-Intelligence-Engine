"""
Download and preparation script for VERITAS-X Phase 1 sample datasets.
Fetches official splits from FEVER and AVeriTeC, and creates a verified multilingual benchmark.
"""
import os
import json
import urllib.request
from typing import Dict, Any, List

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


def ensure_dirs():
    for sub in ["fever", "averitec", "multilingual"]:
        os.makedirs(os.path.join(RAW_DATA_DIR, sub), exist_ok=True)


def download_fever_sample(output_file: str, target_per_class: int = 200) -> int:
    """
    Streams official FEVER shared_task_dev.jsonl and extracts a balanced stratified sample.
    """
    url = "https://fever.ai/download/fever/shared_task_dev.jsonl"
    print(f"[*] Streaming FEVER dev split from official URL: {url} ...")
    
    counts = {"SUPPORTS": 0, "REFUTES": 0, "NOT ENOUGH INFO": 0}
    total_target = target_per_class * 3
    collected = []

    req = urllib.request.Request(url, headers={"User-Agent": "VERITAS-X Research Crawler/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if not line_str:
                continue
            rec = json.loads(line_str)
            lbl = rec.get("label")
            if lbl in counts and counts[lbl] < target_per_class:
                collected.append(line_str)
                counts[lbl] += 1
                if sum(counts.values()) >= total_target:
                    break

    with open(output_file, "w", encoding="utf-8") as out:
        for line in collected:
            out.write(line + "\n")

    print(f"[+] Successfully wrote {len(collected)} balanced FEVER records to {output_file}")
    print(f"    Class distribution: {counts}")
    return len(collected)


def download_averitec_sample(output_file: str) -> int:
    """
    Downloads official AVeriTeC dev split from the official GitHub repository.
    """
    url = "https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/dev.json"
    print(f"[*] Downloading AVeriTeC dev split from official GitHub: {url} ...")

    req = urllib.request.Request(url, headers={"User-Agent": "VERITAS-X Research Crawler/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read().decode("utf-8")
        data = json.loads(content)

    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(data, out, indent=2, ensure_ascii=False)

    print(f"[+] Successfully wrote {len(data)} AVeriTeC records to {output_file}")
    return len(data)


def create_diagnostic_multilingual(output_file: str) -> int:
    """
    Creates a rigorous diagnostic test suite of multilingual claims across
    English, Hindi, Bengali, and Code-Mixed Hinglish, structured per IndicClaimBuster conventions.
    """
    claims = [
        # Hindi (Devanagari)
        {
            "id": "indic_hi_01",
            "claim": "विश्व स्वास्थ्य संगठन ने भारत को पोलियो मुक्त देश घोषित किया है।",
            "language": "hi",
            "label": "SUPPORTS",
            "domain": "Health",
            "claim_date": "2014-03-27",
            "speaker": "WHO Regional Office",
            "evidence": [
                "विश्व स्वास्थ्य संगठन (WHO) ने 27 मार्च 2014 को आधिकारिक रूप से भारत समेत 11 दक्षिण-पूर्व एशियाई देशों को पोलियो मुक्त प्रमाणित किया था।",
                "भारत में अंतिम पोलियो का मामला जनवरी 2011 में पश्चिम बंगाल में दर्ज किया गया था।"
            ]
        },
        {
            "id": "indic_hi_02",
            "claim": "नीति आयोग ने 2024 में भारतीय रुपए पर नई डिजिटल मुद्रा पर प्रतिबंध लगाया।",
            "language": "hi",
            "label": "REFUTES",
            "domain": "Economy",
            "claim_date": "2024-01-15",
            "speaker": "Social Media Rumor",
            "evidence": [
                "भारतीय रिजर्व बैंक (RBI) द्वारा सेंट्रल बैंक डिजिटल करेंसी (e-Rupee) को पायलट प्रोजेक्ट के रूप में जारी रखा गया है, नीति आयोग ने कोई प्रतिबंध नहीं लगाया है।"
            ]
        },
        {
            "id": "indic_hi_03",
            "claim": "तुलसी के पत्ते चबाने से सभी प्रकार के कैंसर 24 घंटे में पूरी तरह ठीक हो जाते हैं।",
            "language": "hi",
            "label": "REFUTES",
            "domain": "Health",
            "claim_date": "2023-08-10",
            "speaker": "WhatsApp Forward",
            "evidence": [
                "आयुर्वेद और चिकित्सा विज्ञान के अनुसार तुलसी रोग प्रतिरोधक क्षमता बढ़ा सकती है, परंतु यह कैंसर का पुख्ता 24 घंटे का इलाज नहीं है।"
            ]
        },
        {
            "id": "indic_hi_04",
            "claim": "चंद्रयान-3 मिशन के प्रज्ञान रोवर ने चंद्रमा के दक्षिणी ध्रुव पर सल्फर की उपस्थिति की पुष्टि की।",
            "language": "hi",
            "label": "SUPPORTS",
            "domain": "Science",
            "claim_date": "2023-08-30",
            "speaker": "ISRO",
            "evidence": [
                "इसरो ने 29 अगस्त 2023 को स्पष्ट किया कि प्रज्ञान रोवर पर लगे LIBS उपकरण ने चंद्रमा के दक्षिणी ध्रुव के निकट सतह पर गंधक (सल्फर) की निर्विवाद उपस्थिति की पुष्टि की।"
            ]
        },
        {
            "id": "indic_hi_05",
            "claim": "हिमाचल प्रदेश में कल रात भारी हिमपात हुआ।",
            "language": "hi",
            "label": "NEI",
            "domain": "Environment",
            "claim_date": "2023-11-05",
            "speaker": "Anonymous Post",
            "evidence": []
        },

        # Bengali (Bangla)
        {
            "id": "indic_bn_01",
            "claim": "রবীন্দ্রনাথ ঠাকুর ১৯১৩ সালে সাহিত্যে নোবেল পুরস্কার পেয়েছিলেন।",
            "language": "bn",
            "label": "SUPPORTS",
            "domain": "Culture",
            "claim_date": "1913-11-13",
            "speaker": "Nobel Prize Committee",
            "evidence": [
                "রবীন্দ্রনাথ ঠাকুর তাঁর গীতাঞ্জলি কাব্যগ্রন্থের জন্য ১৯১৩ সালে প্রথম ভারতীয় তথা এশীয় হিসেবে সাহিত্যে নোবেল পুরস্কার লাভ করেন।"
            ]
        },
        {
            "id": "indic_bn_02",
            "claim": "কলকাতা ভারতের বর্তমান রাজনৈতিক রাজধানী।",
            "language": "bn",
            "label": "REFUTES",
            "domain": "Politics",
            "claim_date": "2023-05-12",
            "speaker": "Unverified Post",
            "evidence": [
                "১৯১১ সালে ভারতের রাজধানী কলকাতা থেকে দিল্লিতে স্থানান্তরিত হয় এবং বর্তমান ভারতের রাজধানী নতুন দিল্লি।"
            ]
        },
        {
            "id": "indic_bn_03",
            "claim": "বিশ্ব স্বাস্থ্য সংস্থা জানিয়েছে যে গরম জল পান করলে কোভিড-১৯ সম্পূর্ণ প্রতিরোধ করা যায়।",
            "language": "bn",
            "label": "REFUTES",
            "domain": "Health",
            "claim_date": "2020-04-18",
            "speaker": "Social Media Viral Video",
            "evidence": [
                "হু (WHO) বারবার নিশ্চিত করেছে যে অতিরিক্ত গরম জল বা চা পান করোনা ভাইরাস ধ্বংস করতে পারে না।"
            ]
        },
        {
            "id": "indic_bn_04",
            "claim": "সুন্দরবন বিশ্বের বৃহত্তম ম্যানগ্রোভ অরণ্য এবং এটি ভারত ও বাংলাদেশের মধ্যে বিস্তৃত।",
            "language": "bn",
            "label": "SUPPORTS",
            "domain": "Environment",
            "claim_date": "2022-09-01",
            "speaker": "UNESCO",
            "evidence": [
                "ইউনেস্কো বিশ্ব ঐতিহ্যবাহী স্থান হিসেবে স্বীকৃত সুন্দরবন প্রায় ১০,০০০ বর্গ কিলোমিটার জুড়ে বিস্তৃত বিশ্বের বৃহত্তম অবিচ্ছিন্ন ম্যানগ্রোভ বন।"
            ]
        },
        {
            "id": "indic_bn_05",
            "claim": "আগামী মাসে দার্জিলিংয়ে নতুন বিমানবন্দর তৈরি হচ্ছে।",
            "language": "bn",
            "label": "NEI",
            "domain": "Infrastructure",
            "claim_date": "2024-02-01",
            "speaker": "Forum Rumor",
            "evidence": []
        },

        # Code-Mixed (Hinglish)
        {
            "id": "indic_hinglish_01",
            "claim": "RBI ne sabhi 2000 rupee notes ko circulation se withdraw karne ka circular issue kiya tha.",
            "language": "hi-en",
            "label": "SUPPORTS",
            "domain": "Economy",
            "claim_date": "2023-05-19",
            "speaker": "Reserve Bank of India",
            "evidence": [
                "On May 19, 2023, the Reserve Bank of India announced the withdrawal of 2,000 denomination banknotes from circulation under Clean Note Policy."
            ]
        },
        {
            "id": "indic_hinglish_02",
            "claim": "Govt ne announce kiya hai ki kal se pure desh mein internet band rahega permanently.",
            "language": "hi-en",
            "label": "REFUTES",
            "domain": "Law & Order",
            "claim_date": "2023-10-12",
            "speaker": "Viral WhatsApp Message",
            "evidence": [
                "Government of India has made no such announcement; national internet shutdowns are illegal and rumors were debunked by PIB Fact Check."
            ]
        },
        {
            "id": "indic_hinglish_03",
            "claim": "Lemon aur baking soda mix karke peene se kidney stones 2 ghante mein dissolve ho jate hain.",
            "language": "hi-en",
            "label": "REFUTES",
            "domain": "Health",
            "claim_date": "2023-07-22",
            "speaker": "Health Influencer Reel",
            "evidence": [
                "Medical nephrologists confirm that while citrate in lemons can prevent certain stones, drinking baking soda does not instantly dissolve kidney stones."
            ]
        },
        {
            "id": "indic_hinglish_04",
            "claim": "ISRO ne Aditya-L1 solar mission ko September 2023 mein successfully launch kiya tha.",
            "language": "hi-en",
            "label": "SUPPORTS",
            "domain": "Science",
            "claim_date": "2023-09-02",
            "speaker": "ISRO Official Press",
            "evidence": [
                "Aditya-L1 was successfully launched by ISRO's PSLV-C57 rocket from Sriharikota on September 2, 2023."
            ]
        },
        {
            "id": "indic_hinglish_05",
            "claim": "Agla match kal Delhi ground par hone wala hai.",
            "language": "hi-en",
            "label": "NEI",
            "domain": "Sports",
            "claim_date": "2023-11-10",
            "speaker": "Anonymous Tweet",
            "evidence": []
        },

        # English (Indic Context)
        {
            "id": "indic_en_01",
            "claim": "India became the fourth country in the world to successfully land a spacecraft on the Moon.",
            "language": "en",
            "label": "SUPPORTS",
            "domain": "Science",
            "claim_date": "2023-08-23",
            "speaker": "Global News / NASA",
            "evidence": [
                "With the landing of Chandrayaan-3 on August 23, 2023, India joined the US, Soviet Union, and China as the fourth nation to achieve a lunar soft landing."
            ]
        },
        {
            "id": "indic_en_02",
            "claim": "The United Nations officially declared the Indian National Anthem as the best anthem in the world.",
            "language": "en",
            "label": "REFUTES",
            "domain": "Politics",
            "claim_date": "2022-08-15",
            "speaker": "Chain Email",
            "evidence": [
                "UNESCO and the UN have repeatedly clarified that they do not rank or award 'best national anthem' recognitions to any country."
            ]
        },
        {
            "id": "indic_en_03",
            "claim": "Drinking warm clove water cures diabetes permanently within seven days.",
            "language": "en",
            "label": "REFUTES",
            "domain": "Health",
            "claim_date": "2023-04-14",
            "speaker": "Facebook Health Post",
            "evidence": [
                "Clinical endocrinology guidelines confirm that Type 1 and Type 2 diabetes are chronic metabolic conditions that cannot be cured permanently with clove water."
            ]
        },
        {
            "id": "indic_en_04",
            "claim": "UPI transactions in India crossed 10 billion monthly transactions in August 2023.",
            "language": "en",
            "label": "SUPPORTS",
            "domain": "Economy",
            "claim_date": "2023-09-01",
            "speaker": "NPCI Official",
            "evidence": [
                "The National Payments Corporation of India (NPCI) reported that Unified Payments Interface (UPI) transactions surpassed 10 billion for the first time in August 2023."
            ]
        },
        {
            "id": "indic_en_05",
            "claim": "A new comet was observed over Chennai yesterday evening.",
            "language": "en",
            "label": "NEI",
            "domain": "Astronomy",
            "claim_date": "2023-12-01",
            "speaker": "Social Media Post",
            "evidence": []
        }
    ]

    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(claims, out, indent=2, ensure_ascii=False)

    print(f"[+] Successfully wrote {len(claims)} multilingual diagnostic records to {output_file}")
    return len(claims)


def main():
    ensure_dirs()
    fever_path = os.path.join(RAW_DATA_DIR, "fever", "sample_dev.jsonl")
    averitec_path = os.path.join(RAW_DATA_DIR, "averitec", "sample_dev.json")
    multilingual_path = os.path.join(RAW_DATA_DIR, "multilingual", "diagnostic_multilingual.json")

    download_fever_sample(fever_path, target_per_class=200)
    download_averitec_sample(averitec_path)
    create_diagnostic_multilingual(multilingual_path)
    print("\n[+] Phase 1 sample dataset acquisition complete!")


if __name__ == "__main__":
    main()

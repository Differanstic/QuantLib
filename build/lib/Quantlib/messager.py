from discord import Webhook
import requests

bot_channel = 'https://discord.com/api/webhooks/1394777477263524021/up0BeAYY5YVdFe99P4pUjo74ohAnIagzf13bP8oyN5dEgFTMkhlc__mXJEuLyn0JJj00'

news_channel = 'https://discord.com/api/webhooks/1405061665657196605/2twxmtejF-YtJ1MECeAnhpy4ZjtY5xtzqPujHxy3f9NAIQ5jfz_gJxzw0W9ekNBAyxUX'

def send_news(content):
    payload = {"content": content  }
    try:
        response = requests.post(news_channel, json=payload)
        response.raise_for_status()
        
    except Exception as e:
        print(f"Failed to send Discord message: {e}")

def send_update_file(file,filename,msg):
    response = requests.post(
        bot_channel,
        files={"file": (filename, file)},
        data={"content": f"📊 {msg}"}
    )
    if response.status_code == 204:
        pass
    else:
        print(f"❌ Failed to send: {response.status_code}\n{response.text}")


def send_update_message(content):
    payload = {"content": content  }
    try:
        response = requests.post(bot_channel, json=payload)
        response.raise_for_status()
        
    except Exception as e:
        print(f"Failed to send Discord message: {e}")

def send_discord_message(content):
    webhook = 'https://discord.com/api/webhooks/1392859080250626138/poWTCPJ82kgNqS5-Qip_N_3FccNn47mkKT42HSuMX3Cydij8ItpZL2w1-20JRuMghpnu'
    payload = {
        "content": content  
    }

    try:
        response = requests.post(webhook, json=payload)
        response.raise_for_status()
       
    except Exception as e:
        print(f"Failed to send Discord message: {e}")
        
def send_telegram_message(text:str):
    bot_token = '7766877299:AAHERYX_ZasL-A1wa_H90k23NMp1iLzoJlY'
    chat_id = '6934209556'
    chat_id = '-4762735986'
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")
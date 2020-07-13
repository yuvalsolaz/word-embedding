
import os,sys
from gtts import gTTS


def main():
    if len(sys.argv) < 2:
        print(f'usage: python {sys.argv[0]} <free text>')
        exit(1)

    text = sys.argv[1]
    print (text)

    speech = gTTS(text = text, lang = 'en', slow = False)
    audion_file = r'text.mp3'
    print(f'saving audio to {audion_file}')
    speech.save(audion_file)

    # os.system(f'cvlc {audion_file}')
    return audion_file

if __name__ == '__main__':
    main()

# Signify
A cross model translation system from Speech to Indian Sign Language along with an emotion recognition system. The model uses [Uberi Speech Recognition](https://github.com/Uberi/speech_recognition) API which takes real-time speech input, convert it to text and then the model generates sign language symbols for that. Used Natural Language Toolkit (NLTK) to add sentiment analysis support which uses 2-way polarity (positive & negative) classification system for the input. The model uses a dictionary-based machine translation-system and the user interface for the project has been designed using EasyGui.

The model consist of a sub-part - [Text-Emotion-Recognition](/Text-Emotion-Detection) which consist of an LSVM model for sentiment analysis. And the trained model is then dumped in the form of a pickel which is later used in the final application so as to avoid training it for every single time. 

## Running the project
- Clone this repository using terminal: `git clone https://github.com/theDeepanshuMourya/Signify`
- Extract the zip file and then open terminal in the Signify folder.
- Now run: `python main.py`
- Now the program will ask for a live voice input.
- Provide an input using microphone and see the results. Happy Translation :thumbsup:

## Contributing
1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :smile:

Check the link in the description for a working model walkthrough.

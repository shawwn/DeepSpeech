import sys
import os
import logging
import argparse
import subprocess
import shlex
import numpy as np
import wavTranscriber

# Debug helpers
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


def main(args):
    parser = argparse.ArgumentParser(description='Transcribe long audio files using webRTC VAD or use the streaming interface')
    parser.add_argument('--aggressive', type=int, choices=range(4), required=False,
                        help='Determines how aggressive filtering out non-speech is. (Interger between 0-3)')
    parser.add_argument('--audio', required=False,
                        help='Path to the audio file to run (WAV format)')
    parser.add_argument('--model', required=True,
                        help='Path to directory that contains all model files (output_graph, lm, trie and alphabet)')
    parser.add_argument('--stream', required=False, action='store_true',
                        help='To use deepspeech streaming interface')
    args = parser.parse_args()
    if args.stream is True and len(sys.argv[1:]) == 3:
             print("Opening mic for streaming")
    elif args.audio is not None and len(sys.argv[1:]) == 6:
            logging.debug("Transcribing audio file @ %s" % args.audio)
    else:
        parser.print_help()
        parser.exit()

    # Point to a path containing the pre-trained models & resolve ~ if used
    dirName = os.path.expanduser(args.model)

    # Resolve all the paths of model files
    output_graph, alphabet, lm, trie = wavTranscriber.resolve_models(dirName)

    # Load output_graph, alpahbet, lm and trie
    model_retval = wavTranscriber.load_model(output_graph, alphabet, lm, trie)

    if args.audio is not None:
        title_names = ['Filename', 'Duration(s)', 'Inference Time(s)', 'Model Load Time(s)', 'LM Load Time(s)']
        print("\n%-30s %-20s %-20s %-20s %s" % (title_names[0], title_names[1], title_names[2], title_names[3], title_names[4]))

        inference_time = 0.0

        # Run VAD on the input file
        waveFile = args.audio
        #segments, sample_rate, audio_length, frames = wavTranscriber.vad_segment_generator(waveFile, args.aggressive)
        #segments, sample_rate, audio_length, frames = wavTranscriber.vad_equal_segment_generator(waveFile, ms=30*9)
        segments, sample_rate, audio_length, frames = wavTranscriber.vad_equal_segment_generator(waveFile, ms=30)
        sctx = model_retval[0].setupStream()
        logging.debug("Saving Transcript @: %s" % waveFile.rstrip(".wav") + ".txt")
        transcript = ''
        rate=16000
        audio_total=0.0

        for i, segment in enumerate(segments):
            # Run deepspeech on the chunk that just completed VAD
            audio = np.frombuffer(segment, dtype=np.int16)
            audio_length = len(audio) * (1 / rate)
            audio_total += audio_length
            #logging.debug("Processing chunk %002d (%0.3fs)" % (i,audio_length))
            #output = wavTranscriber.stt(model_retval[0], audio, sample_rate)
            model_retval[0].feedAudioContent(sctx, audio)
            #if i % (3*9) == 0:
            if i % 3 == 0:
            #if i % 1 == 0:
                #inference_time += output[1]
                transcript2 = model_retval[0].intermediateDecode(sctx)
                if transcript != transcript2:
                    transcript = transcript2
                    logging.debug("Processing chunk %002d (%0.3fs / %002d) Transcript: %s" % (i, audio_length, len(audio), transcript))

        #transcript = model_retval[0].intermediateDecode(sctx)
        transcript = model_retval[0].finishStream(sctx)
        logging.debug("Finished (%0.3fs) Transcript: %s" % (audio_total, transcript))
        with open(waveFile.rstrip(".wav") + ".txt", 'w') as f:
          f.write(transcript + "\n")

        # Summary of the files processed

        # Extract filename from the full file path
        filename, ext = os.path.split(os.path.basename(waveFile))
        logging.debug("************************************************************************************************************")
        logging.debug("%-30s %-20s %-20s %-20s %s" % (title_names[0], title_names[1], title_names[2], title_names[3], title_names[4]))
        logging.debug("%-30s %-20.3f %-20.3f %-20.3f %-0.3f" % (filename + ext, audio_length, inference_time, model_retval[1], model_retval[2]))
        logging.debug("************************************************************************************************************")
        print("%-30s %-20.3f %-20.3f %-20.3f %-0.3f" % (filename + ext, audio_length, inference_time, model_retval[1], model_retval[2]))
    else:
        sctx = model_retval[0].setupStream()
        model_retval[0].feedAudioContent(sctx, np.frombuffer(b'\x00'*512, np.int16))
        subproc = subprocess.Popen(shlex.split('rec -q -V0 -e signed -L -c 1 -b 16 -r 16k -t raw - gain -2'),
                                   stdout=subprocess.PIPE,
                                   bufsize=0)
        print('You can start speaking now. Press Control-C to stop recording.')

        try:
            i = 0
            last_frame = i
            last_total = 0.0
            audio_total = 0.0
            transcript = ' '
            rate = 16000
            while True:
                data = subproc.stdout.read(2048)
                audio = np.frombuffer(data, np.int16)
                audio_length = len(audio) * (1 / rate)
                last_total += audio_length
                audio_total += audio_length
                model_retval[0].feedAudioContent(sctx, audio)
                should_log = False
                if i % 1 == 0:
                    if last_total > 1.5 and not transcript.endswith(' ') and transcript != '':
                        logging.debug('Resetting stream')
                        transcript = model_retval[0].finishStream(sctx)
                        sctx = model_retval[0].setupStream()
                        last_total = 0.0
                        should_log = True
                    transcript2 = model_retval[0].intermediateDecode(sctx)
                    if transcript != transcript2:
                        last_frame = i
                        last_total = 0.0
                        transcript = transcript2
                        should_log = True
                    if int(audio_total) != int(audio_total - audio_length):
                        should_log = True
                if should_log:
                    logging.debug("Processing chunk %002d (%0.3fs / %002d) Transcript: %s" % (i, audio_length, len(audio), repr(transcript)))
                i = i + 1
        except KeyboardInterrupt:
            transcript = model_retval[0].finishStream(sctx)
            sys.stdout.write('Transcription: %s\n\n' % transcript)
            sys.stdout.flush()
            #print('Transcription: ', trans)
            subproc.terminate()
            subproc.wait()


if __name__ == '__main__':
    main(sys.argv[1:])

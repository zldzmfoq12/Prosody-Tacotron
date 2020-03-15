import argparse
import os
import re
import numpy as np
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer
from util import audio

sentences = [
    '피곤해서 쉬고 싶다더니 신나게 얘기하고 있네.',
    '고고와 미미는 열심히 헤엄져 고래 할아버지에게 갔어.',
    '고고 내 힘으로는 빨대를 꺼내지 못할 것 같아.',
    '버려진 플라스틱들이 엄청나게 늘어났어.',
    '하나 언니가 읽어주는 신나는 동화나라.',
]


def get_output_base_path(checkpoint_path):
    base_dir = os.path.dirname(checkpoint_path)
    m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
    name = 'eval-%d' % int(m.group(1)) if m else 'eval'
    return os.path.join(base_dir, name)


def run_eval(args):
    print(hparams_debug_string())
    reference_mel = None
    synth = Synthesizer()
    synth.load(args.checkpoint, args.reference_audio)

    if args.reference_audio is not None:
        ref_wav = audio.load_wav(args.reference_audio)
        reference_mel = audio.melspectrogram(ref_wav).astype(np.float32).T

    base_path = get_output_base_path(args.checkpoint)

    for i, text in enumerate(sentences):
        path = '%s_%d_%.1f_%d.wav' % (base_path+'_gst', hparams.gst_index, hparams.gst_scale, i)
        print('Synthesizing: %s' % path)
        with open(path, 'wb') as f:
            f.write(synth.synthesize(text, reference_mel=reference_mel))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--hparams', default='',
        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--reference_audio', default=None, help='Reference audio path')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    hparams.parse(args.hparams)
    run_eval(args)


if __name__ == '__main__':
  main()

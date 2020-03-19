import argparse
import os
import re
import numpy as np
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer
from util import audio

sentences = [
    '그 사람은 참 예의가 바르다.',
    '매너가 참 좋으시군요.',
    '아직도 그 버릇 못 고쳤니?',
    '요리 실력이 정말 좋으시군요.',
    '그녀는 허스키한 목소리로 노래를 불렀다.',
    '천천히 하세요, 급할 것 없어요',
    '베토벤은 귀가 먼 후에도 작곡을 계속했다.'
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

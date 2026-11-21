from types import SimpleNamespace

import pytest
from PIL import Image

from supra.vibe import SupraVibe, file_hash, main, training_manifest


def test_curated_media_must_match_collection_receipt(tmp_path):
    vibe = pytest.importorskip('particle_sliders_vibe')

    media = tmp_path / 'native.png'
    Image.new('RGB', (256, 256)).save(media)
    output = dict(path=str(media), sha256=file_hash(media))
    case = dict(seed=7, gain=0.75,
                neutral=dict(output=output, judgment=dict(score=0.1)),
                positive=dict(output=output, judgment=dict(score=0.85)))
    ranking = [dict(prompt=dict(neutral='knight', positive='boss knight'),
                    fitness=0.75, cases=[case])]
    collection = dict(identity=dict(schema=vibe.SCHEMA, product=dict(product='supra-particle-sliders')),
                      rounds=[dict(index=0, ranking=ranking)])

    assert training_manifest(collection)['rows'][0]['seed'] == 7
    Image.new('RGB', (256, 256), 'blue').save(media)
    with pytest.raises(ValueError, match='curated Supra media changed'):
        training_manifest(collection)


def test_changed_judge_presentation_is_rejected_before_scoring(tmp_path):
    calls = []

    def score(vibe, output, *, request_id):
        calls.append(request_id)
        with Image.open(output['path']) as image:
            assert image.size == (512, 512)
        return {'score': 1.0}

    product = SupraVibe.__new__(SupraVibe)
    product.workflows = SimpleNamespace(score=score)
    product.score_directory = tmp_path / 'judgments'
    native = tmp_path / 'native.png'
    Image.new('RGB', (256, 256)).save(native)
    output = {'path': str(native)}

    presentation = product.score('boss', output, request_id='first')['presentation']
    assert product.score('boss', output, request_id='repeat')['presentation'] == presentation
    Image.new('RGB', (512, 512), 'blue').save(presentation['presented'])
    with pytest.raises(ValueError, match='judge presentation differs'):
        product.score('boss', output, request_id='tampered')
    assert calls == ['first', 'repeat']


def test_invalid_min_score_is_rejected_before_collection(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(['--vibe', 'boss', '--out', str(tmp_path), '--min-score', 'nan'])
    assert error.value.code == 2
    assert '--min-score must be in [0,1]' in capsys.readouterr().err

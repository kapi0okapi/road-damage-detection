import cv2
import numpy as np

from road_damage.datasets.rdd2022 import block_split, convert, parse_voc

XML = """<annotation><filename>{name}.jpg</filename>
<size><width>600</width><height>600</height><depth>3</depth></size>
<object><name>D40</name><bndbox><xmin>100</xmin><ymin>200</ymin><xmax>200</xmax><ymax>260</ymax></bndbox></object>
<object><name>D00</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>590</ymax></bndbox></object>
<object><name>Repair</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object>
</annotation>"""


def make_country(root, country, n):
    xml_dir = root / "RDD2022" / country / "train" / "annotations" / "xmls"
    img_dir = root / "RDD2022" / country / "train" / "images"
    xml_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)
    for i in range(n):
        name = f"{country}_{i:06d}"
        (xml_dir / f"{name}.xml").write_text(XML.format(name=name))
        cv2.imwrite(str(img_dir / f"{name}.jpg"), np.zeros((8, 8, 3), np.uint8))


def test_parse_voc_keeps_four_classes(tmp_path):
    (tmp_path / "a.xml").write_text(XML.format(name="a"))
    w, h, boxes = parse_voc(tmp_path / "a.xml")
    assert (w, h) == (600, 600) and [b[0] for b in boxes] == [3, 0]
    np.testing.assert_allclose(boxes[0][1:], [0.25, 0.383333, 0.166667, 0.1], atol=1e-5)


def test_block_split_proportions():
    split = block_split(1000)
    assert abs((split == "test").mean() - 0.15) < 0.01 and abs((split == "val").mean() - 0.15) < 0.01


def test_convert_writes_lists_and_experiment_yamls(tmp_path):
    make_country(tmp_path, "Norway", 40)
    make_country(tmp_path, "Japan", 40)
    stats = convert(tmp_path, tmp_path / "out")
    out = tmp_path / "out"
    assert stats["Norway"]["images"]["train"] == 28 and stats["Norway"]["boxes"]["D40"] == 40
    for name in ("all", "no_norway", "test_Norway", "norway_10", "norway_100"):
        assert (out / "data" / f"{name}.yaml").exists()
    assert "Norway" not in (out / "data" / "no_norway.yaml").read_text()
    first = (out / "Norway_train.txt").read_text().split()[0]
    assert first.startswith("./images/") and (out / first).exists()
    assert (out / "labels" / f"{first.split('/')[-1][:-4]}.txt").read_text().count("\n") == 2
    assert len((out / "Norway_train_10.txt").read_text().split()) == 2
    replay = (out / "Others_replay.txt").read_text().split()
    assert len(replay) == 28 and all("Japan" in line for line in replay)
    assert "Japan_val.txt" in (out / "data" / "norway_replay.yaml").read_text()

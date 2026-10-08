from pretrain.train_tokenizer import mixed_sample


def test_the_sample_gives_nasa_its_share_of_the_characters():
    nasa = "N" * 100
    web = "\n\n".join(["w" * 49] * 20)  # 20 documents of 49 characters + 2-character breaks
    sample = mixed_sample(nasa, web, nasa_share=0.25)
    assert sample.startswith(nasa)
    assert 390 <= len(sample) <= 410  # 100 is 25% of 400


def test_the_web_part_ends_at_a_document_break_not_mid_word():
    nasa = "N" * 100
    web = "\n\n".join(["first doc", "second doc", "third doc"] * 50)
    sample = mixed_sample(nasa, web, nasa_share=0.25)
    assert sample.endswith(("first doc", "second doc", "third doc"))

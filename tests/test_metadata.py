from memory_system.metadata import SharedMetadata

def test_metadata_register_and_docs():
    md = SharedMetadata()
    md.register_model("foo", "1.0", {"task": "regression"})
    md.register_feature("x1", "float")
    md.register_subscriber("svc-A")
    doc = md.generate_documentation()
    assert "foo:1.0" in doc and "x1" in doc and "svc-A" in doc
    assert len(md.revision_history()) >= 1\n
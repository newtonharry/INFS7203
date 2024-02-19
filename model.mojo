from python import Python


fn main() raises:
    Python.add_to_path("./")
    let model = Python.import_module("model")
    model.run()

import torch as t
from torch.utils.data import DataLoader, TensorDataset
from utils import report, allclose, allclose_scalar, assert_all_equal


@report
def test_mlp(MLP):
    import w1d4_part1_solution

    x = t.randn(128, 2)
    t.manual_seed(534)
    mlp = MLP(2, 32, 2)

    params = list(mlp.parameters())
    if not params:
        raise AssertionError(
            "Your network doesn't return anything when mlp.parameters() is called. If you used a list for your layers, you need to use nn.ModuleList or nn.Sequential instead."
        )

    t.manual_seed(534)
    _mlp = w1d4_part1_solution.ImageMemorizer(2, 32, 2)

    allclose(mlp(x), _mlp(x))


@report
def test_tensor_dataset(TensorDataset):
    tensors = [t.rand((10, 20)), t.rand((10, 5)), t.arange(10)]
    dataset = TensorDataset(*tensors)
    assert len(dataset) == 10
    for index in [0, slice(0, 5, 1), slice(1, 5, 2)]:
        print("Testing with index:", index)
        expected = tuple(tensor[index] for tensor in tensors)
        actual = dataset[index]
        for e, a in zip(expected, actual):
            assert_all_equal(e, a)


@report
def test_train(train_one_epoch):
    import w1d4_part1_solution

    t.manual_seed(928)
    lr = 0.1
    momentum = 0.5
    X = t.rand(512, 2)
    Y = t.rand(512, 3)
    dl = DataLoader(TensorDataset(X, Y), batch_size=128)

    t.manual_seed(600)
    expected_model = w1d4_part1_solution.ImageMemorizer(2, 32, 3)
    expected_train_loss = w1d4_part1_solution.train_one_epoch(expected_model, dl)

    t.manual_seed(600)
    actual_model = w1d4_part1_solution.ImageMemorizer(2, 32, 3)
    actual_train_loss = train_one_epoch(actual_model, dl)

    allclose_scalar(actual_train_loss, expected_train_loss)

    x = t.randn(128, 2)
    allclose(
        actual_model(x.to(w1d4_part1_solution.device)),
        expected_model(x.to(w1d4_part1_solution.device)),
    )


@report
def test_evaluate(evaluate):
    import w1d4_part1_solution

    # TBD: loss on random labels? On predict all zero? Not sure how good it is to do this.
    t.manual_seed(928)
    X = t.rand(512, 2)
    Y = t.rand(512, 3)
    dl = DataLoader(TensorDataset(X, Y), batch_size=128)

    model = w1d4_part1_solution.ImageMemorizer(2, 32, 3)
    w1d4_part1_solution.train_one_epoch(model, dl)
    _loss = w1d4_part1_solution.evaluate(model, dl)
    loss = evaluate(model, dl)
    allclose_scalar(_loss, loss)

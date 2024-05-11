#include "tests.h"
#include "Network.h"

using namespace NeuralNets;

namespace {

    void AFConstructorTest() {
        auto af_sigmoid = ActivationFunction::Initialize(AF_id::Sigmoid);
        auto af_relu = ActivationFunction::Initialize(AF_id::ReLu);
        auto af_tanh = ActivationFunction::Initialize(AF_id::Tanh);
    }

    void SigApplyTest() {
        auto af = ActivationFunction::Initialize(AF_id::Sigmoid);
        Matrix mat(3, 4);
        mat << 12, 3, 2, 1, 3, 17, 3, 10, 17, 7, 14, 2;
        mat = af.Apply(mat);
        Matrix res(3, 4);
        res << 0.999994, 0.952574, 0.880797, 0.731059,
                0.952574, 1, 0.952574, 0.999955,
                1, 0.999089, 0.999999, 0.880797;
        assert(abs((mat - res).array().maxCoeff()) < 1e-5);
    }

    void SigDeriveTest() {
        auto af = ActivationFunction::Initialize(AF_id::Sigmoid);
        Matrix mat(3, 4);
        mat << 12, 3, 2, 1, 3, 17, 3, 10, 17, 7, 14, 2;
        mat = af.Derivative(mat);
        Matrix res(3, 4);
        res << 6.14414e-06, 0.0451767, 0.104994, 0.196612,
                0.0451767, 4.13994e-08, 0.0451767, 4.53958e-05,
                4.13994e-08, 0.000910221, 8.31527e-07, 0.104994;
        assert(abs((mat - res).array().maxCoeff()) < 1e-5);
    }

    void ReLuApplyTest() {
        auto af = ActivationFunction::Initialize(AF_id::ReLu);
        Matrix mat(3, 4);
        mat << -15, -12, 0, 13, 14, -2, -7, 3, -2, 2, -7, 11;
        mat = af.Apply(mat);
        Matrix res(3, 4);
        res << 0, 0, 0, 13, 14, 0, 0, 3, 0, 2, 0, 11;
        assert(abs((mat - res).array().maxCoeff()) < 1e-5);
    }

    void ReLuDeriveTest() {
        auto af = ActivationFunction::Initialize(AF_id::ReLu);
        Matrix mat(3, 4);
        mat << -15, -12, 0, 13, 14, -2, -7, 3, -2, 2, -7, 11;
        mat = af.Derivative(mat);
        Matrix res(3, 4);
        res << 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1;
        assert(abs((mat - res).array().maxCoeff()) < 1e-5);
    }

    void TanhApplyTest() {
        auto af = ActivationFunction::Initialize(AF_id::Tanh);
        Matrix mat(3, 4);
        mat << -2, 7, 13, 3, 5, 6, -3, 9, -4, 17, 4, 2;
        mat = af.Apply(mat);
        Matrix res(3, 4);
        res << -0.964028, 0.999998, 1, 0.995055, 0.999909, 0.999988, -0.995055, 1, -0.999329, 1, 0.999329, 0.964028;
        assert(abs((mat - res).array().maxCoeff()) < 1e-5);
    }

    void TanhDeriveTest() {
        auto af = ActivationFunction::Initialize(AF_id::Tanh);
        Matrix mat(3, 4);
        mat << -2, 7, 13, 3, 5, 6, -3, 9, -4, 17, 4, 2;
        mat = af.Derivative(mat);
        Matrix res(3, 4);
        res << 0.0706508, 3.32611e-06, 2.04364e-11, 0.00986604, 0.000181583, 2.45765e-05, 0.00986604, 6.09199e-08,
                0.00134095, 6.85563e-15, 0.00134095, 0.0706508;
        assert(abs((mat - res).array().maxCoeff()) < 1e-5);
    }

    void AFIsEmptyTest() {
        auto af = ActivationFunction::Initialize(AF_id::Sigmoid);
        af.IsEmpty();
    }

    void LayerConstructorTest() {
        auto layer = Layer(2, 3, AF_id::Sigmoid);
    }

    void LayerEvaluateTest() {
        auto layer = Layer(2, 3, AF_id::ReLu);
        Matrix batch(2, 4);
        batch << 2, 3, 1, 4, 5, 6, 8, 7;
        batch = layer.Evaluate(batch);
        Matrix res(3, 4);
        res << 4.48878, 5.85235, 5.99677, 7.21592, 2.1214, 3.03562, 2.8912, 3.94983, 0, 0, 0, 0;
        assert(abs((batch - res).array().maxCoeff()) < 1e-5);
    }

    void LayerGradATest() {
        auto layer = Layer(2, 3, AF_id::ReLu);
        Matrix batch(2, 4);
        batch << 2, 3, 1, 4, 5, 6, 8, 7;
        Matrix grad(4, 3);
        grad << 3, 7, 6, 2, 1, -2, 5, 9, 6, 10, -4, 5;
        batch = layer.GradA(batch, grad);
        Matrix res(3, 2);
        res << 14.25, 34.25, 0, 0, 0, 0;
        assert(abs((batch - res).array().maxCoeff()) < 1e-5);
    }

    void LayerGradbTest() {
        auto layer = Layer(2, 3, AF_id::Sigmoid);
        Matrix batch(2, 4);
        batch << 2, 3, 1, 4, 5, 6, 8, 7;
        Matrix grad(4, 3);
        grad << 3, 7, 6, 2, 1, -2, 5, 9, 6, 10, -4, 5;
        batch = layer.Gradb(batch, grad);
        Vector res(3);
        res << 0.0696423, 0.513158, 0.914589;
        assert(abs((batch - res).array().maxCoeff()) < 1e-5);
    }

    void LayerChangeParamsTest() {
        auto layer = Layer(2, 3, AF_id::Tanh);
        Matrix delta_A(3, 2);
        delta_A << 2, 4, 5, 1, 7, 8;
        Vector delta_b(3);
        delta_b << 5, 2, 7;
        layer.ChangeParams(delta_A, delta_b, 3);
    }

    void LayerNextGradTest() {
        auto layer = Layer(2, 3, AF_id::Tanh);
        Matrix batch(2, 4);
        batch << 2, 3, 1, 4, 5, 6, 8, 7;
        Matrix grad(4, 3);
        grad << 3, 7, 6, 2, 1, -2, 5, 9, 6, 10, -4, 5;
        batch = layer.NextGrad(batch, grad);
        Matrix res(4, 2);
        res << 1.70035, -1.09931, 1.07194, -0.760499, 0.361157, -0.246008, 4.82832, -3.36471;
        assert(abs((batch - res).array().maxCoeff()) < 1e-5);
    }

    void RandTest() {
        auto rand = Rand::GetNormal(3, 2);
        Matrix res(3, 2);
        res << 0.222999, -0.405438, -0.215125, 0.680288, -0.467574, -0.952513;
        assert(abs((rand - res).array().maxCoeff()) < 1e-5);
    }

    void PenaltyConstructorTest() {
        auto penalty_mse = Penalty::Initialize(PenaltyId::MSE);
        auto penalty_mae = Penalty::Initialize(PenaltyId::MAE);
        auto penalty_ent = Penalty::Initialize(PenaltyId::CrossEntropy);
    }

    void CalcPenaltyMSETest() {
        auto penalty = Penalty::Initialize(PenaltyId::MSE);
        Matrix batch(3, 2);
        batch << 2, 7, 7, 8, 1, 4;
        Matrix net_out(3, 2);
        net_out << 3, 5, 6, 9, 0, 3;
        IndexType2 res = 4.5;
        assert(abs(penalty.CalcPenalty(net_out, batch) - res) < 1e-5);
    }

    void FindInitialGradientMSETest() {
        auto penalty = Penalty::Initialize(PenaltyId::MSE);
        Matrix batch(3, 2);
        batch << 2, 7, 7, 8, 1, 4;
        Matrix net_out(3, 2);
        net_out << 3, 5, 6, 9, 0, 3;
        batch = penalty.FindInitialGradient(net_out, batch);
        Matrix res(2, 3);
        res << 2, -2, -2, -4, 2, -2;
        assert(abs((batch - res).array().maxCoeff()) < 1e-5);
    }

    void CalcPenaltyMAETest() {
        auto penalty = Penalty::Initialize(PenaltyId::MAE);
        Matrix batch(3, 2);
        batch << 2, 7, 7, 8, 1, 4;
        Matrix net_out(3, 2);
        net_out << 3, 5, 6, 9, 0, 3;
        IndexType2 res = 3.5;
        assert(abs(penalty.CalcPenalty(net_out, batch) - res) < 1e-5);
    }

    void FindInitialGradientMAETest() {
        auto penalty = Penalty::Initialize(PenaltyId::MAE);
        Matrix batch(3, 2);
        batch << 2, 7, 7, 8, 1, 4;
        Matrix net_out(3, 2);
        net_out << 3, 5, 6, 9, 0, 3;
        batch = penalty.FindInitialGradient(net_out, batch);
        Matrix res(2, 3);
        res << 1, -1, -1, -1, 1, -1;
        assert(abs((batch - res).array().maxCoeff()) < 1e-5);
    }

    void CalcPenaltyEntTest() {
        auto penalty = Penalty::Initialize(PenaltyId::CrossEntropy);
        Matrix batch(3, 2);
        batch << 2, 7, 7, 0, 1, 4;
        Matrix net_out(3, 2);
        net_out << 3, 5, 6, 9, 2, 3;
        IndexType2 res = 15.5466;
        assert(abs(penalty.CalcPenalty(net_out, batch) - res) < 1e-5);
    }

    void FindInitialGradientEntTest() {
        auto penalty = Penalty::Initialize(PenaltyId::CrossEntropy);
        Matrix batch(3, 2);
        batch << 2, 7, 7, 0, 1, 4;
        Matrix net_out(3, 2);
        net_out << 3, 5, 6, 9, 2, 3;
        batch = penalty.FindInitialGradient(net_out, batch);
        Matrix res(2, 3);
        res << -0.666667, -1.16667, -0.5, -1.4, 0, -1.33333;
        assert(abs((batch - res).array().maxCoeff()) < 1e-5);
    }

    void PenaltyIsEmptyTest() {
        auto penalty = Penalty::Initialize(PenaltyId::MAE);
        penalty.IsEmpty();
    }

    void LearningRateConstructorTest() {
        LearningRate lr_const(1.0);
        LearningRate lr_linear(3.5, LRSchedule::Linear);
        LearningRate lr_exp(2.0, 4.5);
    }

    void LearningRateConstGetRateTest() {
        LearningRate lr_const(1.0);
        IndexType2 res = 1.0;
        assert(abs(lr_const.GetRate(25) - res) < 1e-5);
    }

    void LearningRateLinearGetRateTest() {
        LearningRate lr_linear(3.5, LRSchedule::Linear);
        IndexType2 res = 0.14;
        assert(abs(lr_linear.GetRate(25) - res) < 1e-5);
    }

    void LearningRateExpGetRateTest() {
        LearningRate lr_exp(2.0, 4.5);
        IndexType2 res = 7.93687e-69;
        assert(abs(lr_exp.GetRate(35) - res) < 1e-5);
    }

    void LearningRateChangeRateTest() {
        LearningRate lr(2.0);
        lr.ChangeRate();
        IndexType2 res = 0.0740741;
        assert(abs(lr.GetRate(27) - res) < 1e-5);
    }

    void NetworkConstructorTest() {
        auto net = Network({2, 4, 3, 5}, {AF_id::Sigmoid, AF_id::ReLu, AF_id::Tanh});
    }

    void NetworkEvaluateTest() {
        auto net = Network({2, 4, 3}, {AF_id::Sigmoid, AF_id::ReLu});
        Matrix batch(2, 4);
        batch << 2, 7, 7, 0, 1, 4, 9, 1;
        Matrix res(3, 4);
        res << 0, 0, 0, 0, 0.537186, 0.609244, 0.85456, 0.564624, 0.65172, 0.638404, 0.541333, 0.627946;
        assert(abs((net.EvaluateNet(batch) - res).array().maxCoeff()) < 1e-5);
    }

    void NetworkTrainTest() {
        auto net = Network({2, 3, 5, 4}, {AF_id::Tanh, AF_id::Sigmoid, AF_id::Tanh});
        Matrix data_x(2, 3);
        Matrix data_y(4, 3);
        LearningRate lr_const(1);
        net.Train(data_x, data_y, 1, Penalty::Initialize(PenaltyId::MSE),
                  0.1, lr_const, 300);
    }
}

void test::RunAllTests() {
    AFConstructorTest();
    SigApplyTest();
    SigDeriveTest();
    ReLuApplyTest();
    ReLuDeriveTest();
    TanhApplyTest();
    TanhDeriveTest();
    AFIsEmptyTest();

    LayerConstructorTest();
    LayerEvaluateTest();
    LayerGradATest();
    LayerGradbTest();
    LayerNextGradTest();
    LayerChangeParamsTest();

    RandTest();

    PenaltyConstructorTest();
    CalcPenaltyMSETest();
    FindInitialGradientMSETest();
    CalcPenaltyMAETest();
    FindInitialGradientMAETest();
    CalcPenaltyEntTest();
    FindInitialGradientEntTest();
    PenaltyIsEmptyTest();

    LearningRateConstructorTest();
    LearningRateConstGetRateTest();
    LearningRateLinearGetRateTest();
    LearningRateExpGetRateTest();
    LearningRateChangeRateTest();

    NetworkConstructorTest();
    NetworkEvaluateTest();
    NetworkTrainTest();
}
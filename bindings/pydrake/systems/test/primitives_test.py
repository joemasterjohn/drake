import copy
import gc
import scipy.sparse
import unittest
import numpy as np

from pydrake.autodiffutils import AutoDiffXd
from pydrake.common import RandomDistribution, RandomGenerator
from pydrake.common.test_utilities import numpy_compare
from pydrake.common.test_utilities.deprecation import catch_drake_warnings
from pydrake.common.value import Value
from pydrake.symbolic import Expression, Variable
from pydrake.systems.framework import (
    BasicVector, BasicVector_,
    BusValue,
    DiagramBuilder,
    DiagramBuilder_,
    InputPort,
    TriggerType,
    VectorBase,
    kUseDefaultName,
)
from pydrake.systems.test.test_util import (
    MyVector2,
)
from pydrake.systems.primitives import (
    Adder, Adder_,
    AddRandomInputs,
    AffineSystem, AffineSystem_,
    BusCreator, BusCreator_,
    BusSelector, BusSelector_,
    ConstantValueSource, ConstantValueSource_,
    ConstantVectorSource, ConstantVectorSource_,
    ControllabilityMatrix,
    Demultiplexer, Demultiplexer_,
    DiscreteDerivative, DiscreteDerivative_,
    DiscreteTimeDelay, DiscreteTimeDelay_,
    DiscreteTimeIntegrator_,
    FirstOrderLowPassFilter,
    FirstOrderTaylorApproximation,
    Gain, Gain_,
    Integrator, Integrator_,
    IsControllable,
    IsDetectable,
    IsObservable,
    IsStabilizable,
    Linearize,
    LinearSystem, LinearSystem_,
    LinearTransformDensity, LinearTransformDensity_,
    LogVectorOutput,
    MatrixGain,
    Multiplexer, Multiplexer_,
    MultilayerPerceptron, MultilayerPerceptron_,
    ObservabilityMatrix,
    PassThrough, PassThrough_,
    PerceptronActivationType,
    PortSwitch, PortSwitch_,
    RandomSource,
    Saturation, Saturation_,
    Selector, Selector_, SelectorParams,
    SharedPointerSystem, SharedPointerSystem_,
    Sine, Sine_,
    SparseMatrixGain_,
    StateInterpolatorWithDiscreteDerivative,
    StateInterpolatorWithDiscreteDerivative_,
    SymbolicVectorSystem, SymbolicVectorSystem_,
    TrajectoryAffineSystem, TrajectoryAffineSystem_,
    TrajectoryLinearSystem, TrajectoryLinearSystem_,
    TrajectorySource, TrajectorySource_,
    VectorLog, VectorLogSink, VectorLogSink_,
    WrapToSystem, WrapToSystem_,
    ZeroOrderHold, ZeroOrderHold_,
)
from pydrake.trajectories import PiecewisePolynomial


def compare_value(test, a, b):
    # Compares a vector or abstract value.
    if isinstance(a, VectorBase):
        test.assertTrue(np.allclose(a.get_value(), b.get_value()))
    else:
        test.assertEqual(type(a.get_value()), type(b.get_value()))
        test.assertEqual(a.get_value(), b.get_value())


class TestGeneral(unittest.TestCase):
    def _check_instantiations(self, template, supports_symbolic=True):
        default_cls = template[None]
        self.assertTrue(template[float] is default_cls)
        self.assertTrue(template[AutoDiffXd] is not default_cls)
        if supports_symbolic:
            self.assertTrue(template[Expression] is not default_cls)

    def test_instantiations(self):
        # TODO(eric.cousineau): Refine tests once NumPy functionality is
        # resolved for dtype=object, or dtype=custom is used.
        self._check_instantiations(Adder_)
        self._check_instantiations(AffineSystem_)
        self._check_instantiations(BusCreator_)
        self._check_instantiations(BusSelector_)
        self._check_instantiations(ConstantValueSource_)
        self._check_instantiations(ConstantVectorSource_)
        self._check_instantiations(Demultiplexer_)
        self._check_instantiations(DiscreteDerivative_)
        self._check_instantiations(DiscreteTimeDelay_)
        self._check_instantiations(DiscreteTimeIntegrator_)
        self._check_instantiations(Gain_)
        self._check_instantiations(Integrator_)
        self._check_instantiations(LinearSystem_)
        self._check_instantiations(LinearTransformDensity_,
                                   supports_symbolic=False)
        self._check_instantiations(Multiplexer_)
        self._check_instantiations(MultilayerPerceptron_)
        self._check_instantiations(PassThrough_)
        self._check_instantiations(PortSwitch_)
        self._check_instantiations(Saturation_)
        self._check_instantiations(Selector_)
        self._check_instantiations(SharedPointerSystem_)
        self._check_instantiations(Sine_)
        self._check_instantiations(StateInterpolatorWithDiscreteDerivative_)
        self._check_instantiations(SymbolicVectorSystem_)
        self._check_instantiations(TrajectoryAffineSystem_,
                                   supports_symbolic=False)
        self._check_instantiations(TrajectoryLinearSystem_,
                                   supports_symbolic=False)
        self._check_instantiations(TrajectorySource_)
        self._check_instantiations(VectorLogSink_)
        self._check_instantiations(WrapToSystem_)
        self._check_instantiations(ZeroOrderHold_)

    @numpy_compare.check_all_types
    def test_discrete_time_integrator(self, T):
        time_step = 0.1
        integrator = DiscreteTimeIntegrator_[T](size=2, time_step=time_step)
        self.assertEqual(integrator.time_step(), time_step)
        context = integrator.CreateDefaultContext()
        x = np.array([1., 2.])
        integrator.set_integral_value(context=context, value=x)
        u = np.array([3., 4.])
        integrator.get_input_port(0).FixValue(context, u)
        x_next = integrator.EvalUniquePeriodicDiscreteUpdate(
            context).get_vector()._get_value_copy()
        numpy_compare.assert_float_equal(x_next, x + time_step * u)

    def test_linear_affine_system(self):
        # Just make sure linear system is spelled correctly.
        A = np.identity(2)
        B = np.array([[0], [1]])
        f0 = np.array([[0], [0]])
        C = np.array([[0, 1]])
        D = [1]
        y0 = [0]
        system = LinearSystem(A, B, C, D)
        context = system.CreateDefaultContext()
        self.assertEqual(system.get_input_port(0).size(), 1)
        self.assertEqual(context
                         .get_mutable_continuous_state_vector().size(), 2)
        self.assertEqual(system.get_output_port(0).size(), 1)
        self.assertTrue((system.A() == A).all())
        self.assertTrue((system.B() == B).all())
        self.assertTrue((system.f0() == f0).all())
        self.assertTrue((system.C() == C).all())
        self.assertEqual(system.D(), D)
        self.assertEqual(system.y0(), y0)
        self.assertEqual(system.time_period(), 0.)

        x0 = np.array([1, 2])
        system.configure_default_state(x0=x0)
        system.SetDefaultContext(context)
        np.testing.assert_equal(
            context.get_continuous_state_vector().CopyToVector(), x0)
        generator = RandomGenerator()
        system.SetRandomContext(context, generator)
        np.testing.assert_equal(
            context.get_continuous_state_vector().CopyToVector(), x0)
        system.configure_random_state(covariance=np.eye(2))
        system.SetRandomContext(context, generator)
        self.assertNotEqual(
            context.get_continuous_state_vector().CopyToVector()[1], x0[1])

        Co = ControllabilityMatrix(system)
        self.assertEqual(Co.shape, (2, 2))
        self.assertFalse(IsControllable(system))
        self.assertFalse(IsControllable(system, 1e-6))
        self.assertFalse(IsStabilizable(sys=system))
        self.assertFalse(IsStabilizable(sys=system, threshold=1e-6))
        Ob = ObservabilityMatrix(system)
        self.assertEqual(Ob.shape, (2, 2))
        self.assertFalse(IsObservable(system))
        self.assertFalse(IsDetectable(sys=system))
        self.assertFalse(IsDetectable(sys=system, threshold=1e-6))

        system = AffineSystem(A, B, f0, C, D, y0, .1)
        self.assertEqual(system.get_input_port(0), system.get_input_port())
        self.assertEqual(system.get_output_port(0), system.get_output_port())
        context = system.CreateDefaultContext()
        self.assertEqual(system.get_input_port(0).size(), 1)
        self.assertEqual(context.get_discrete_state_vector().size(), 2)
        self.assertEqual(system.get_output_port(0).size(), 1)
        self.assertTrue((system.A() == A).all())
        self.assertTrue((system.B() == B).all())
        self.assertTrue((system.f0() == f0).all())
        self.assertTrue((system.C() == C).all())
        self.assertEqual(system.D(), D)
        self.assertEqual(system.y0(), y0)
        self.assertEqual(system.time_period(), .1)

        system.get_input_port(0).FixValue(context, 0)
        linearized = Linearize(system, context)
        self.assertTrue((linearized.A() == A).all())
        taylor = FirstOrderTaylorApproximation(system, context)
        self.assertTrue((taylor.y0() == y0).all())

        new_A = np.array([[1, 2], [3, 4]])
        new_B = np.array([[5], [6]])
        new_f0 = np.array([[7], [8]])
        new_C = np.array([[9, 10]])
        new_D = np.array([[11]])
        new_y0 = np.array([12])
        system.UpdateCoefficients(
            A=new_A, B=new_B, f0=new_f0, C=new_C, D=new_D, y0=new_y0
        )
        np.testing.assert_equal(new_A, system.A())
        np.testing.assert_equal(new_B, system.B())
        np.testing.assert_equal(new_f0.flatten(), system.f0())
        np.testing.assert_equal(new_C, system.C())
        np.testing.assert_equal(new_D, system.D())
        np.testing.assert_equal(new_y0, system.y0())

        system = MatrixGain(D=A)
        self.assertTrue((system.D() == A).all())

        system = TrajectoryAffineSystem(
            PiecewisePolynomial(A),
            PiecewisePolynomial(B),
            PiecewisePolynomial(f0),
            PiecewisePolynomial(C),
            PiecewisePolynomial(D),
            PiecewisePolynomial(y0),
            .1)
        self.assertEqual(system.get_input_port(0), system.get_input_port())
        self.assertEqual(system.get_output_port(0), system.get_output_port())
        context = system.CreateDefaultContext()
        self.assertEqual(system.get_input_port(0).size(), 1)
        self.assertEqual(context.get_discrete_state_vector().size(), 2)
        self.assertEqual(system.get_output_port(0).size(), 1)
        for t in np.linspace(0., 1., 5):
            self.assertTrue((system.A(t) == A).all())
            self.assertTrue((system.B(t) == B).all())
            self.assertTrue((system.f0(t) == f0).all())
            self.assertTrue((system.C(t) == C).all())
            self.assertEqual(system.D(t), D)
            self.assertEqual(system.y0(t), y0)
        self.assertEqual(system.time_period(), .1)
        x0 = np.array([1, 2])
        system.configure_default_state(x0=x0)
        system.SetDefaultContext(context)
        np.testing.assert_equal(
            context.get_discrete_state_vector().CopyToVector(), x0)
        generator = RandomGenerator()
        system.SetRandomContext(context, generator)
        np.testing.assert_equal(
            context.get_discrete_state_vector().CopyToVector(), x0)
        system.configure_random_state(covariance=np.eye(2))
        system.SetRandomContext(context, generator)
        self.assertNotEqual(
            context.get_discrete_state_vector().CopyToVector()[1], x0[1])

        system = TrajectoryLinearSystem(
            A=PiecewisePolynomial(A),
            B=PiecewisePolynomial(B),
            C=PiecewisePolynomial(C),
            D=PiecewisePolynomial(D),
            time_period=0.1)
        self.assertEqual(system.time_period(), .1)
        system.configure_default_state(x0=np.array([1, 2]))
        system.configure_random_state(covariance=np.eye(2))

    def test_linear_affine_system_empty_matrices(self):
        # Confirm the default values for the system matrices in the
        # constructor.
        def CheckSizes(system, num_states, num_inputs, num_outputs):
            self.assertEqual(system.num_continuous_states(), num_states)
            self.assertEqual(system.num_inputs(), num_inputs)
            self.assertEqual(system.num_outputs(), num_outputs)

        # A constant vector system.
        system = AffineSystem(y0=[2, 1])
        CheckSizes(system, num_states=0, num_inputs=0, num_outputs=2)

        # A matrix gain.
        system = AffineSystem(D=np.eye(2))
        CheckSizes(system, num_states=0, num_inputs=2, num_outputs=2)
        system = LinearSystem(D=np.eye(2))
        CheckSizes(system, num_states=0, num_inputs=2, num_outputs=2)

        # Add an offset.
        system = AffineSystem(D=np.eye(2), y0=[1, 2])
        CheckSizes(system, num_states=0, num_inputs=2, num_outputs=2)

        # An integrator.
        system = LinearSystem(B=np.eye(2))
        CheckSizes(system, num_states=2, num_inputs=2, num_outputs=0)

    def test_linear_system_zero_size(self):
        # Explicitly test #12633.
        num_x = 0
        num_y = 2
        num_u = 2
        A = np.zeros((num_x, num_x))
        B = np.zeros((num_x, num_u))
        C = np.zeros((num_y, num_x))
        D = np.zeros((num_y, num_u))
        self.assertIsNotNone(LinearSystem(A, B, C, D))

    @numpy_compare.check_nonsymbolic_types
    def test_linear_transform_density(self, T):
        dut = LinearTransformDensity_[T](
            distribution=RandomDistribution.kGaussian,
            input_size=3,
            output_size=3)
        w_in = np.array([T(0.5), T(0.1), T(1.5)])
        context = dut.CreateDefaultContext()
        dut.get_input_port_w_in().FixValue(context, w_in)
        self.assertEqual(dut.get_input_port_A().size(), 9)
        self.assertEqual(dut.get_input_port_b().size(), 3)
        self.assertEqual(dut.get_distribution(), RandomDistribution.kGaussian)
        A = np.array([
            [T(0.5), T(1), T(2)], [T(1), T(2), T(3)], [T(3), T(4), T(5)]])
        dut.FixConstantA(context=context, A=A)
        b = np.array([T(1), T(2), T(3)])
        dut.FixConstantB(context=context, b=b)

        dut.CalcDensity(context=context)

        self.assertEqual(dut.get_output_port_w_out().size(), 3)
        self.assertEqual(dut.get_output_port_w_out_density().size(), 1)

    @numpy_compare.check_all_types
    def test_bus_creator(self, T):
        dut = BusCreator_[T](output_port_name="foo")
        dut.DeclareVectorInputPort(name="vec", size=2)
        dut.DeclareVectorInputPort(name=kUseDefaultName, size=2)
        dut.DeclareAbstractInputPort(name="abst", model_value=Value[str]())
        dut.DeclareAbstractInputPort(name=kUseDefaultName,
                                     model_value=Value[object]())
        # Check exactly what types show up on the output bus.
        context = dut.CreateDefaultContext()
        dut.GetInputPort("vec").FixValue(context, value=np.ones(2))
        dut.GetInputPort("u1").FixValue(context, value=np.zeros(2))
        dut.GetInputPort("abst").FixValue(context, value="hello")
        dut.GetInputPort("u3").FixValue(context, value=tuple([1, 2, 3]))
        output = dut.get_output_port().Eval(context)
        self.assertIsInstance(output.Find("vec"), BasicVector_[T])
        self.assertIsInstance(output.Find("u1"), BasicVector_[T])
        self.assertIsInstance(output.Find("abst"), str)
        self.assertIsInstance(output.Find("u3"), tuple)

    @numpy_compare.check_all_types
    def test_bus_selector(self, T):
        dut = BusSelector_[T](input_port_name="bar")
        dut.DeclareVectorOutputPort(name="vec", size=2)
        dut.DeclareVectorOutputPort(name=kUseDefaultName, size=2)
        dut.DeclareAbstractOutputPort(name="abst", model_value=Value[str]())
        dut.DeclareAbstractOutputPort(name=kUseDefaultName,
                                      model_value=Value[object]())
        # Make sure we know exactly how to populate a bus-valued input port.
        input_bus = BusValue()
        vec = np.ones(2)
        y1 = np.zeros(2)
        abst = "hello"
        y3 = tuple([1, 2, 3])
        input_bus.Set("vec", Value(BasicVector_[T](vec)))
        input_bus.Set("y1", Value(BasicVector_[T](y1)))
        input_bus.Set("abst", Value("hello"))
        input_bus.Set("y3", Value(y3))
        context = dut.CreateDefaultContext()
        dut.get_input_port().FixValue(context, input_bus)
        numpy_compare.assert_float_equal(
            dut.GetOutputPort("vec").Eval(context), vec)
        numpy_compare.assert_float_equal(
            dut.GetOutputPort("y1").Eval(context), y1)
        self.assertEqual(dut.GetOutputPort("abst").Eval(context), abst)
        self.assertEqual(dut.GetOutputPort("y3").Eval(context), y3)

    def test_vector_pass_through(self):
        model_value = BasicVector([1., 2, 3])
        system = PassThrough(vector_size=model_value.size())
        context = system.CreateDefaultContext()
        system.get_input_port(0).FixValue(context, model_value)
        output = system.AllocateOutput()
        input_eval = system.EvalVectorInput(context, 0)
        compare_value(self, input_eval, model_value)
        system.CalcOutput(context, output)
        output_value = output.get_vector_data(0)
        compare_value(self, output_value, model_value)

    def test_default_vector_pass_through(self):
        model_value = [1., 2, 3]
        system = PassThrough(value=model_value)
        context = system.CreateDefaultContext()
        np.testing.assert_array_equal(
            model_value, system.get_output_port().Eval(context))

    def test_abstract_pass_through(self):
        model_value = Value("Hello world")
        system = PassThrough(abstract_model_value=model_value)
        context = system.CreateDefaultContext()
        system.get_input_port(0).FixValue(context, model_value)
        output = system.AllocateOutput()
        input_eval = system.EvalAbstractInput(context, 0)
        compare_value(self, input_eval, model_value)
        system.CalcOutput(context, output)
        output_value = output.get_data(0)
        compare_value(self, output_value, model_value)

    def test_port_switch(self):
        system = PortSwitch(vector_size=2)
        a = system.DeclareInputPort(name="a")
        system.DeclareInputPort(name="b")
        context = system.CreateDefaultContext()
        self.assertIsInstance(a, InputPort)
        system.get_port_selector_input_port().FixValue(context, a.get_index())

    def test_first_order_low_pass_filter(self):
        filter1 = FirstOrderLowPassFilter(time_constant=3.0, size=4)
        self.assertEqual(filter1.get_time_constant(), 3.0)

        alpha = np.array([1, 2, 3])
        filter2 = FirstOrderLowPassFilter(time_constants=alpha)
        np.testing.assert_array_equal(filter2.get_time_constants_vector(),
                                      alpha)

        context = filter2.CreateDefaultContext()
        filter2.set_initial_output_value(context, [0., -0.2, 0.4])

    def test_gain(self):
        k = 42.
        input_size = 10
        systems = [Gain(k=k, size=input_size),
                   Gain(k=k*np.ones(input_size))]

        for system in systems:
            context = system.CreateDefaultContext()
            output = system.AllocateOutput()

            def mytest(input, expected):
                system.get_input_port(0).FixValue(context, input)
                system.CalcOutput(context, output)
                self.assertTrue(np.allclose(output.get_vector_data(
                    0).CopyToVector(), expected))

            test_input = np.arange(input_size)
            mytest(np.arange(input_size), k*np.arange(input_size))

    def test_integrator(self):
        n = 3
        initial_value = np.array((1.0, 2.0, 3.0))
        size_integrator = Integrator(size=n)
        value_integrator = Integrator(initial_value=initial_value)
        size_context = size_integrator.CreateDefaultContext()
        value_context = value_integrator.CreateDefaultContext()
        self.assertTrue(np.array_equal(
            size_integrator.get_output_port(0).Eval(size_context),
            (0.0, 0.0, 0.0)))
        self.assertTrue(np.array_equal(
            value_integrator.get_output_port(0).Eval(value_context),
            initial_value))
        value_integrator.set_default_integral_value(2 * initial_value)
        value_context2 = value_integrator.CreateDefaultContext()
        self.assertTrue(np.array_equal(
            value_integrator.get_output_port(0).Eval(value_context2),
            2 * initial_value))
        size_integrator.set_integral_value(context=size_context,
                                           value=initial_value)
        self.assertTrue(np.array_equal(
            size_integrator.get_output_port(0).Eval(size_context),
            initial_value))

    def test_saturation(self):
        system = Saturation((0., -1., 3.), (1., 2., 4.))
        context = system.CreateDefaultContext()
        output = system.AllocateOutput()

        def mytest(input, expected):
            system.get_input_port(0).FixValue(context, input)
            system.CalcOutput(context, output)
            self.assertTrue(np.allclose(output.get_vector_data(
                0).CopyToVector(), expected))

        mytest((-5., 5., 4.), (0., 2., 4.))
        mytest((.4, 0., 3.5), (.4, 0., 3.5))

    def _make_selector_params(self):
        def _select(i, j):
            return SelectorParams.OutputSelection(
                input_port_index=i,
                input_offset=j,
            )
        return SelectorParams(
            inputs=[
                SelectorParams.InputPortParams(name="x", size=3),
                SelectorParams.InputPortParams(name="y", size=2),
                SelectorParams.InputPortParams(name="z", size=1),
            ],
            outputs=[
                # a = <x[0], x[1]>
                SelectorParams.OutputPortParams(
                    name="a",
                    selections=[_select(0, 0), _select(0, 1)],
                ),
                # b = <x[2], y[0]>
                SelectorParams.OutputPortParams(
                    name="b",
                    selections=[_select(0, 2), _select(1, 0)],
                ),
                # c = <y[1], z[0]>
                SelectorParams.OutputPortParams(
                    name="c",
                    selections=[_select(1, 1), _select(2, 0)]
                ),
            ],
        )

    def test_selector_params(self):
        dut = self._make_selector_params()
        self.assertEqual(len(dut.inputs), 3)
        copy.deepcopy(dut)
        self.maxDiff = None
        self.assertEqual(
            repr(dut),
            "SelectorParams("
            "inputs=["
            "InputPortParams(name='x', size=3), "
            "InputPortParams(name='y', size=2), "
            "InputPortParams(name='z', size=1)], "
            "outputs=["
            "OutputPortParams(name='a', selections=["
            "OutputSelection(input_port_index=0, input_offset=0), "
            "OutputSelection(input_port_index=0, input_offset=1)]), "
            "OutputPortParams(name='b', selections=["
            "OutputSelection(input_port_index=0, input_offset=2), "
            "OutputSelection(input_port_index=1, input_offset=0)]), "
            "OutputPortParams(name='c', selections=["
            "OutputSelection(input_port_index=1, input_offset=1), "
            "OutputSelection(input_port_index=2, input_offset=0)])])")

    @numpy_compare.check_all_types
    def test_selector(self, T):
        dut = Selector_[T](params=self._make_selector_params())
        dut.CreateDefaultContext()

    def test_trajectory_source(self):
        ppt = PiecewisePolynomial.FirstOrderHold(
            [0., 1.], [[2., 3.], [2., 1.]])
        system = TrajectorySource(trajectory=ppt,
                                  output_derivative_order=0,
                                  zero_derivatives_beyond_limits=True)
        context = system.CreateDefaultContext()
        output = system.AllocateOutput()

        def mytest(input, expected):
            context.SetTime(input)
            system.CalcOutput(context, output)
            self.assertTrue(np.allclose(output.get_vector_data(
                0).CopyToVector(), expected))

        mytest(0.0, (2.0, 2.0))
        mytest(0.5, (2.5, 1.5))
        mytest(1.0, (3.0, 1.0))

        ppt2 = PiecewisePolynomial.FirstOrderHold(
            [0., 1.], [[4., 6.], [4., 2.]])
        system.UpdateTrajectory(trajectory=ppt2)
        mytest(0.0, (4.0, 4.0))
        mytest(0.5, (5.0, 3.0))
        mytest(1.0, (6.0, 2.0))

    def test_symbolic_vector_system(self):
        t = Variable("t")
        x = [Variable("x0"), Variable("x1")]
        u = [Variable("u0"), Variable("u1")]
        system = SymbolicVectorSystem(time=t, state=x, input=u,
                                      dynamics=[x[0] + x[1], t],
                                      output=[u[1]],
                                      time_period=0.0)
        context = system.CreateDefaultContext()

        self.assertEqual(context.num_continuous_states(), 2)
        self.assertEqual(context.num_discrete_state_groups(), 0)
        self.assertEqual(system.get_input_port(0).size(), 2)
        self.assertEqual(system.get_output_port(0).size(), 1)
        self.assertEqual(context.num_abstract_parameters(), 0)
        self.assertEqual(context.num_numeric_parameter_groups(), 0)
        self.assertTrue(system.dynamics_for_variable(x[0])
                        .EqualTo(x[0] + x[1]))
        self.assertTrue(system.dynamics_for_variable(x[1])
                        .EqualTo(t))

    def test_symbolic_vector_system_parameters(self):
        t = Variable("t")
        x = [Variable("x0"), Variable("x1")]
        u = [Variable("u0"), Variable("u1")]
        p = [Variable("p0"), Variable("p1")]
        system = SymbolicVectorSystem(time=t, state=x, input=u,
                                      parameter=p,
                                      dynamics=[p[0] * x[0] + x[1] + p[1], t],
                                      output=[u[1]],
                                      time_period=0.0)
        context = system.CreateDefaultContext()

        self.assertEqual(context.num_continuous_states(), 2)
        self.assertEqual(context.num_discrete_state_groups(), 0)
        self.assertEqual(system.get_input_port(0).size(), 2)
        self.assertEqual(system.get_output_port(0).size(), 1)
        self.assertEqual(context.num_abstract_parameters(), 0)
        self.assertEqual(context.num_numeric_parameter_groups(), 1)
        self.assertEqual(context.get_numeric_parameter(0).size(), 2)
        self.assertTrue(system.dynamics_for_variable(x[0])
                        .EqualTo(p[0] * x[0] + x[1] + p[1]))
        self.assertTrue(system.dynamics_for_variable(x[1])
                        .EqualTo(t))

    def test_wrap_to_system(self):
        system = WrapToSystem(2)
        system.set_interval(1, 1., 2.)
        context = system.CreateDefaultContext()
        output = system.AllocateOutput()

        def mytest(input, expected):
            system.get_input_port(0).FixValue(context, input)
            system.CalcOutput(context, output)
            self.assertTrue(np.allclose(output.get_vector_data(
                0).CopyToVector(), expected))

        mytest((-1.5, 0.5), (-1.5, 1.5))
        mytest((.2, .3), (.2, 1.3))

    def test_demultiplexer(self):
        # Test demultiplexer with scalar outputs.
        demux = Demultiplexer(size=4)
        context = demux.CreateDefaultContext()
        self.assertEqual(demux.num_input_ports(), 1)
        self.assertEqual(demux.num_output_ports(), 4)
        numpy_compare.assert_equal(demux.get_output_ports_sizes(),
                                   [1, 1, 1, 1])

        input_vec = np.array([1., 2., 3., 4.])
        demux.get_input_port(0).FixValue(context, input_vec)
        output = demux.AllocateOutput()
        demux.CalcOutput(context, output)

        for i in range(4):
            self.assertTrue(
                np.allclose(output.get_vector_data(i).get_value(),
                            input_vec[i]))

        # Test demultiplexer with vector outputs.
        demux = Demultiplexer(size=4, output_ports_size=2)
        context = demux.CreateDefaultContext()
        self.assertEqual(demux.num_input_ports(), 1)
        self.assertEqual(demux.num_output_ports(), 2)
        numpy_compare.assert_equal(demux.get_output_ports_sizes(), [2, 2])

        demux.get_input_port(0).FixValue(context, input_vec)
        output = demux.AllocateOutput()
        demux.CalcOutput(context, output)

        for i in range(2):
            self.assertTrue(
                np.allclose(output.get_vector_data(i).get_value(),
                            input_vec[2*i:2*i+2]))

        # Test demultiplexer with different output port sizes.
        output_ports_sizes = np.array([1, 2, 1])
        num_output_ports = output_ports_sizes.size
        input_vec = np.array([1., 2., 3., 4.])
        demux = Demultiplexer(output_ports_sizes=output_ports_sizes)
        context = demux.CreateDefaultContext()
        self.assertEqual(demux.num_input_ports(), 1)
        self.assertEqual(demux.num_output_ports(), num_output_ports)
        numpy_compare.assert_equal(demux.get_output_ports_sizes(),
                                   output_ports_sizes)

        demux.get_input_port(0).FixValue(context, input_vec)
        output = demux.AllocateOutput()
        demux.CalcOutput(context, output)

        output_port_start = 0
        for i in range(num_output_ports):
            output_port_size = output.get_vector_data(i).size()
            self.assertTrue(
                np.allclose(output.get_vector_data(i).get_value(),
                            input_vec[output_port_start:
                                      output_port_start+output_port_size]))
            output_port_start += output_port_size

    def test_multiplexer(self):
        my_vector = MyVector2(data=[1., 2.])
        test_cases = [
            dict(has_vector=False, mux=Multiplexer(num_scalar_inputs=4),
                 data=[[5.], [3.], [4.], [2.]]),
            dict(has_vector=False, mux=Multiplexer(input_sizes=[2, 3]),
                 data=[[8., 4.], [3., 6., 9.]]),
            dict(has_vector=True, mux=Multiplexer(model_vector=my_vector),
                 data=[[42.], [3.]]),
        ]
        for case in test_cases:
            mux = case['mux']
            port_size = sum([len(vec) for vec in case['data']])
            self.assertEqual(mux.get_output_port(0).size(), port_size)
            context = mux.CreateDefaultContext()
            output = mux.AllocateOutput()
            num_ports = len(case['data'])
            self.assertEqual(context.num_input_ports(), num_ports)
            for j, vec in enumerate(case['data']):
                mux.get_input_port(j).FixValue(context, vec)
            mux.CalcOutput(context, output)
            self.assertTrue(
                np.allclose(output.get_vector_data(0).get_value(),
                            [elem for vec in case['data'] for elem in vec]))
            if case['has_vector']:
                # Check the type matches MyVector2.
                value = output.get_vector_data(0)
                self.assertTrue(isinstance(value, MyVector2))

    def test_multilayer_perceptron(self):
        mlp = MultilayerPerceptron(
            layers=[1, 2, 3], activation_type=PerceptronActivationType.kReLU)
        self.assertEqual(mlp.get_input_port().size(), 1)
        self.assertEqual(mlp.get_output_port().size(), 3)
        context = mlp.CreateDefaultContext()
        params = np.zeros((mlp.num_parameters(), 1))
        self.assertEqual(mlp.num_parameters(), 13)
        self.assertEqual(mlp.layers(), [1, 2, 3])
        self.assertEqual(mlp.activation_type(layer=0),
                         PerceptronActivationType.kReLU)
        self.assertEqual(len(mlp.GetParameters(context=context)),
                         mlp.num_parameters())
        mlp.SetWeights(context=context, layer=0, W=np.array([[1], [2]]))
        mlp.SetBiases(context=context, layer=0, b=[3, 4])
        np.testing.assert_array_equal(
            mlp.GetWeights(context=context, layer=0), np.array([[1], [2]]))
        np.testing.assert_array_equal(
            mlp.GetBiases(context=context, layer=0), np.array([3, 4]))
        params = np.zeros(mlp.num_parameters())
        mlp.SetWeights(params=params, layer=0, W=np.array([[1], [2]]))
        mlp.SetBiases(params=params, layer=0, b=[3, 4])
        np.testing.assert_array_equal(
            mlp.GetWeights(params=params, layer=0), np.array([[1], [2]]))
        np.testing.assert_array_equal(
            mlp.GetBiases(params=params, layer=0), np.array([3, 4]))
        mutable_params = mlp.GetMutableParameters(context=context)
        mutable_params[:] = 3.0
        np.testing.assert_array_equal(mlp.GetParameters(context),
                                      np.full(mlp.num_parameters(), 3.0))

        global called_loss
        called_loss = False

        def silly_loss(Y, dloss_dY):
            global called_loss
            called_loss = True
            # We must be careful to update the dloss in place, rather than bind
            # a new matrix to the same variable name.
            dloss_dY[:] = 1
            # dloss_dY = np.array(...etc...)  # <== wrong
            return Y.sum()

        dloss_dparams = np.zeros((13,))
        generator = RandomGenerator(23)
        mlp.SetRandomContext(context, generator)
        mlp.Backpropagation(context=context,
                            X=np.array([1, 3, 4]).reshape((1, 3)),
                            loss=silly_loss,
                            dloss_dparams=dloss_dparams)
        self.assertTrue(called_loss)
        self.assertTrue(dloss_dparams.any())  # No longer all zero.

        dloss_dparams = np.zeros((13,))
        mlp.BackpropagationMeanSquaredError(context=context,
                                            X=np.array([1, 3, 4]).reshape(
                                                (1, 3)),
                                            Y_desired=np.eye(3),
                                            dloss_dparams=dloss_dparams)
        self.assertTrue(dloss_dparams.any())  # No longer all zero.

        Y = np.asfortranarray(np.eye(3))
        mlp.BatchOutput(context=context, X=np.array([[0.1, 0.3, 0.4]]), Y=Y)
        self.assertFalse(np.allclose(Y, np.eye(3)))
        Y2 = mlp.BatchOutput(context=context, X=np.array([[0.1, 0.3, 0.4]]))
        np.testing.assert_array_equal(Y, Y2)

        mlp2 = MultilayerPerceptron(layers=[3, 2, 1],
                                    activation_types=[
                                        PerceptronActivationType.kReLU,
                                        PerceptronActivationType.kTanh
        ])
        self.assertEqual(mlp2.activation_type(0),
                         PerceptronActivationType.kReLU)
        self.assertEqual(mlp2.activation_type(1),
                         PerceptronActivationType.kTanh)
        Y = np.asfortranarray(np.full((1, 3), 2.4))
        dYdX = np.asfortranarray(np.full((3, 3), 5.3))
        context2 = mlp2.CreateDefaultContext()
        mlp2.BatchOutput(context=context2, X=np.eye(3), Y=Y, dYdX=dYdX)
        # The default context sets the weights and biases to zero, so the
        # output (and gradients) should be zero.
        np.testing.assert_array_almost_equal(Y, np.zeros((1, 3)))
        np.testing.assert_array_almost_equal(dYdX, np.zeros((3, 3)))

        mlp = MultilayerPerceptron(use_sin_cos_for_input=[True, False],
                                   remaining_layers=[3, 2],
                                   activation_types=[
                                       PerceptronActivationType.kReLU,
                                       PerceptronActivationType.kTanh
        ])
        self.assertEqual(mlp.get_input_port().size(), 2)
        np.testing.assert_array_equal(mlp.layers(), [3, 3, 2])

    def test_random_source(self):
        source = RandomSource(distribution=RandomDistribution.kUniform,
                              num_outputs=2, sampling_interval_sec=0.01)
        self.assertEqual(source.get_output_port(0).size(), 2)

        builder = DiagramBuilder()
        # Note: There are no random inputs to add to the empty diagram, but it
        # confirms the API works.
        AddRandomInputs(sampling_interval_sec=0.01, builder=builder)

        builder_ad = DiagramBuilder_[AutoDiffXd]()
        AddRandomInputs(sampling_interval_sec=0.01, builder=builder_ad)

    def test_constant_vector_source(self):
        source = ConstantVectorSource(source_value=[1., 2.])
        context = source.CreateDefaultContext()
        source.get_source_value(context)
        source.get_mutable_source_value(context)

    def test_ctor_api(self):
        """Tests construction of systems for systems whose executions semantics
        are not tested above.
        """
        ConstantValueSource(Value("Hello world"))
        DiscreteTimeDelay(update_sec=0.1, delay_time_steps=5, vector_size=2)
        DiscreteTimeDelay(
            update_sec=0.1, delay_time_steps=5,
            abstract_model_value=Value("Hello world"))

        ZeroOrderHold(period_sec=0.1, offset_sec=0.0, vector_size=2)
        dut = ZeroOrderHold(period_sec=1.0, offset_sec=0.25,
                            abstract_model_value=Value("Hello world"))
        self.assertEqual(dut.period(), 1.0)
        self.assertEqual(dut.offset(), 0.25)

    def test_shared_pointer_system_ctor(self):
        dut = SharedPointerSystem(value_to_hold=[1, 2, 3])
        readback = dut.get()
        self.assertListEqual(readback, [1, 2, 3])
        del dut
        self.assertListEqual(readback, [1, 2, 3])

    def test_shared_pointer_system_builder(self):
        builder = DiagramBuilder()
        self.assertListEqual(
            SharedPointerSystem.AddToBuilder(
                builder=builder, value_to_hold=[1, 2, 3]),
            [1, 2, 3])
        diagram = builder.Build()
        del builder
        readback = diagram.GetSystems()[0].get()
        self.assertListEqual(readback, [1, 2, 3])
        del diagram
        self.assertListEqual(readback, [1, 2, 3])

    def test_sine(self):
        # Test scalar output.
        sine_source = Sine(amplitude=1, frequency=2, phase=3,
                           size=1, is_time_based=True)
        self.assertEqual(sine_source.get_output_port(0).size(), 1)
        self.assertEqual(sine_source.get_output_port(1).size(), 1)
        self.assertEqual(sine_source.get_output_port(2).size(), 1)

        # Test vector output.
        sine_source = Sine(amplitude=1, frequency=2, phase=3,
                           size=3, is_time_based=True)
        self.assertEqual(sine_source.get_output_port(0).size(), 3)
        self.assertEqual(sine_source.get_output_port(1).size(), 3)
        self.assertEqual(sine_source.get_output_port(2).size(), 3)

        sine_source = Sine(amplitudes=np.ones(2), frequencies=np.ones(2),
                           phases=np.ones(2), is_time_based=True)
        self.assertEqual(sine_source.get_output_port(0).size(), 2)
        self.assertEqual(sine_source.get_output_port(1).size(), 2)
        self.assertEqual(sine_source.get_output_port(2).size(), 2)

    def test_discrete_derivative(self):
        discrete_derivative = DiscreteDerivative(num_inputs=5, time_step=0.5)
        self.assertEqual(discrete_derivative.get_input_port(0).size(), 5)
        self.assertEqual(discrete_derivative.get_output_port(0).size(), 5)
        self.assertEqual(discrete_derivative.time_step(), 0.5)
        self.assertTrue(discrete_derivative.suppress_initial_transient())

        discrete_derivative = DiscreteDerivative(
            num_inputs=5, time_step=0.5, suppress_initial_transient=False)
        self.assertFalse(discrete_derivative.suppress_initial_transient())

    @numpy_compare.check_all_types
    def test_sparse_matrix_gain(self, T):
        D = scipy.sparse.csc_matrix(
            (np.array([2, 1., 3]), np.array([0, 1, 0]),
             np.array([0, 2, 2, 3])), shape=(2, 3))
        dut = SparseMatrixGain_[T](D=D)
        context = dut.CreateDefaultContext()
        u = np.array([1, 2, 3])
        dut.get_input_port().FixValue(context, u)
        y = dut.get_output_port().Eval(context)
        numpy_compare.assert_float_equal(y, D.todense() @ u)

        numpy_compare.assert_float_equal(D.todense(), dut.D().todense())
        D2 = scipy.sparse.csc_matrix(
            (np.array([1, 4, 6]), np.array([0, 1, 0]),
             np.array([0, 2, 2, 3])), shape=(2, 3))
        dut.set_D(D=D2)
        numpy_compare.assert_float_equal(D2.todense(), dut.D().todense())

        # Make sure empty matrices work as expected.
        D00 = scipy.sparse.csc_matrix(([], [], []), shape=(0, 0))
        # Having zero rows doesn't work yet...
        # D01 = scipy.sparse.csc_matrix(([], [], [0]), shape=(0, 1))
        D10 = scipy.sparse.csc_matrix(([1.2], [0], [0]), shape=(1, 0))
        for D in [D00, D10]:
            dut = SparseMatrixGain_[T](D=D)
            context = dut.CreateDefaultContext()
            u = np.ones((D.shape[1], 1))
            dut.get_input_port().FixValue(context, u)
            y = dut.get_output_port().Eval(context)
            self.assertEqual(y.size, D.shape[0])

    def test_state_interpolator_with_discrete_derivative(self):
        state_interpolator = StateInterpolatorWithDiscreteDerivative(
            num_positions=5, time_step=0.4)
        self.assertEqual(state_interpolator.get_input_port(0).size(), 5)
        self.assertEqual(state_interpolator.get_output_port(0).size(), 10)
        self.assertTrue(state_interpolator.suppress_initial_transient())

        # test set_initial_position using context
        context = state_interpolator.CreateDefaultContext()
        state_interpolator.set_initial_position(
            context=context, position=5*[1.1])
        np.testing.assert_array_equal(
            context.get_discrete_state(0).CopyToVector(),
            np.array(5*[1.1]))
        np.testing.assert_array_equal(
            context.get_discrete_state(1).CopyToVector(),
            np.array(5*[1.1]))

        # test set_initial_position using state
        context = state_interpolator.CreateDefaultContext()
        state_interpolator.set_initial_position(
            state=context.get_state(), position=5*[1.3])
        np.testing.assert_array_equal(
            context.get_discrete_state(0).CopyToVector(),
            np.array(5*[1.3]))
        np.testing.assert_array_equal(
            context.get_discrete_state(1).CopyToVector(),
            np.array(5*[1.3]))

        state_interpolator = StateInterpolatorWithDiscreteDerivative(
            num_positions=5, time_step=0.4, suppress_initial_transient=True)
        self.assertTrue(state_interpolator.suppress_initial_transient())

    @numpy_compare.check_nonsymbolic_types
    def test_log_vector_output(self, T):
        # Add various redundant loggers to a system, to exercise the
        # LogVectorOutput bindings.
        builder = DiagramBuilder_[T]()
        kSize = 1
        integrator = builder.AddSystem(Integrator_[T](kSize))
        port = integrator.get_output_port(0)
        loggers = []
        loggers.append(LogVectorOutput(port, builder))
        loggers.append(LogVectorOutput(src=port, builder=builder))
        loggers.append(LogVectorOutput(port, builder, 0.125))
        loggers.append(LogVectorOutput(
            src=port, builder=builder, publish_period=0.125))

        loggers.append(LogVectorOutput(port, builder, {TriggerType.kForced}))
        loggers.append(LogVectorOutput(
            src=port, builder=builder, publish_triggers={TriggerType.kForced}))
        loggers.append(LogVectorOutput(
            port, builder, {TriggerType.kPeriodic}, 0.125))
        loggers.append(LogVectorOutput(
            src=port, builder=builder,
            publish_triggers={TriggerType.kPeriodic}, publish_period=0.125))

        # Check the returned loggers by calling some trivial methods.
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        self.assertTrue(all(logger.FindLog(context).num_samples() == 0
                            for logger in loggers))

    @numpy_compare.check_nonsymbolic_types
    def test_vector_log(self, T):
        kSize = 1
        dut = VectorLog(kSize)
        self.assertEqual(dut.get_input_size(), kSize)
        dut.AddData(0.1, [22.22])
        self.assertEqual(dut.num_samples(), 1)
        self.assertEqual(dut.sample_times(), [0.1])
        self.assertEqual(dut.data(), [22.22])
        dut.Clear()
        self.assertEqual(dut.num_samples(), 0)
        # There is no good way from python to test the semantics of Reserve(),
        # but test the binding anyway.
        dut.Reserve(VectorLog.kDefaultCapacity * 3)

    @numpy_compare.check_nonsymbolic_types
    def test_vector_log_sink(self, T):
        # Add various redundant loggers to a system, to exercise the
        # VectorLog constructor bindings.
        builder = DiagramBuilder_[T]()
        kSize = 1
        constructors = [VectorLogSink_[T]]
        loggers = []
        if T == float:
            constructors.append(VectorLogSink)
        for constructor in constructors:
            loggers.append(builder.AddSystem(constructor(kSize)))
            loggers.append(builder.AddSystem(constructor(input_size=kSize)))
            loggers.append(builder.AddSystem(constructor(kSize, 0.125)))
            loggers.append(builder.AddSystem(
                constructor(input_size=kSize, publish_period=0.125)))
            loggers.append(builder.AddSystem(
                constructor(kSize, {TriggerType.kForced})))
            loggers.append(builder.AddSystem(
                constructor(input_size=kSize,
                            publish_triggers={TriggerType.kForced})))
            loggers.append(builder.AddSystem(
                constructor(kSize, {TriggerType.kPeriodic}, 0.125)))
            loggers.append(builder.AddSystem(
                constructor(input_size=kSize,
                            publish_triggers={TriggerType.kPeriodic},
                            publish_period=0.125)))

        # Exercise all of the log access methods.
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        # FindLog and FindMutableLog find the same object.
        self.assertTrue(
            all(logger.FindLog(context) == logger.FindMutableLog(context)
                for logger in loggers))
        # Build a list of pairs of loggers and their local contexts.
        loggers_and_contexts = [(x, x.GetMyContextFromRoot(context))
                                for x in loggers]
        # GetLog and GetMutableLog find the same object.
        self.assertTrue(
            all(logger.GetLog(logger_context)
                == logger.GetMutableLog(logger_context)
                for logger, logger_context in loggers_and_contexts))
        # GetLog and FindLog find the same object, given the proper contexts.
        self.assertTrue(
            all(logger.GetLog(logger_context) == logger.FindLog(context)
                for logger, logger_context in loggers_and_contexts))

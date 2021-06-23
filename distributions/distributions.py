import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import wikipedia
import functools

distr_list = list()
distr_func_list = list()

# Use decorators to register and select distributions


def register_distr(_func=None, name=None):
    def decorator(func):
        # st.write(name)
        distr_list.append(name)
        # st.write(func.__name__)
        distr_func_list.append(func)
        # The @functools.wraps decorator uses the function functools.update_wrapper() to update special
        # attributes like __name__ and __doc__ that are used in the introspection.
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # print("Something is happening before the function is called.")
            return func(*args, **kwargs)
            # print("Something is happening after the function is called.")
        return wrapper
    if _func is None:
        return decorator  # 2
    else:
        return decorator(_func)  # 3


@register_distr(name='Gaussian')
def dist_gaussian():
    st.subheader('Description')
    # descr = wikipedia.summary('Normal distribution', sentences=8)
    descr = "In probability theory, a normal (or Gaussian or Gauss or Laplace–Gauss) distribution is a type of continuous " \
            "probability distribution for a real-valued random variable.  A random variable with a Gaussian distribution " \
            "is said to be normally distributed and is called a normal deviate. Normal distributions are important in " \
            "statistics and are often used in the natural and social sciences to represent real-valued random " \
            "variables whose distributions are not known.  Gaussian distributions have some unique properties that " \
            "are valuable in analytic studies. For instance, any linear combination of a fixed collection of normal " \
            "deviates is a normal deviate. Many results and methods (such as propagation of uncertainty and least " \
            "squares parameter fitting) can be derived analytically in explicit form when the relevant variables " \
            "are normally distributed."
    st.write(descr)
    st.write('The probability density function for Gaussian is')

    st.latex(r'f(x|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x - \mu)^2}{2\sigma} \right)')

    st.markdown('for a real number $x$ with support on the whole $\mathbb{R}$.')

    # (A) let the user seletect the parameters
    param_mean = st.slider('Location', -10.0, +10.0, 0.0, step=0.01)
    param_var = st.slider('Scale', 0.01, 10.0, 1.0, step=0.01)

    l01 = stats.norm.ppf(0.01, loc=param_mean, scale=param_var)
    l99 = stats.norm.ppf(0.99, loc=param_mean, scale=param_var)

    # (B) generate data based on parameters
    x = np.linspace(-10.0, +10.0, 1000)
    data_pdf = stats.norm.pdf(x, loc=param_mean, scale=param_var)
    data_cdf = stats.norm.cdf(x, loc=param_mean, scale=param_var)

    # (C) plot some nice graphics
    xticks = np.zeros(21+1)
    xticks[1:] = np.linspace(-10., +10., 21)
    xticks_str = ['']*xticks.size
    xticks_str[1] = '{0:1.1f}'.format(xticks[1])
    xticks_str[-1] = '{0:1.1f}'.format(xticks[-1])
    xticks_str[0] = '0'

    yticks = np.linspace(0., 1., 10)
    yticks_str = ['']*yticks.size
    yticks_str[0] = '{0:1.1f}'.format(yticks[0])
    yticks_str[-1] = '{0:1.1f}'.format(yticks[-1])

    st.markdown('Following figures show the PDF (red) and CDF (blue) as well as the 1 and the 99 percentile (black dashed lines). ')
    fig = plt.figure(figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')
    # PDF
    plt.subplot(1, 2, 1)
    plt.plot(x, data_pdf, '-r')
    plt.fill_between(x, np.zeros(x.size), data_pdf, facecolor='red', interpolate=True, alpha=0.4)
    plt.axvline(l01, linestyle='--', color='k', linewidth=0.8)
    plt.axvline(l99, linestyle='--', color='k', linewidth=0.8)
    plt.grid(True, linestyle=':')
    plt.ylabel('PDF')
    plt.xticks(xticks, '')
    plt.yticks(yticks, yticks_str)
    # plt.ylim([0., 1.])
    plt.xlim([-10.1, 10.1])

    # CDF
    plt.subplot(1, 2, 2)
    plt.plot(x, data_cdf, '-b')
    plt.fill_between(x, np.zeros(x.size), data_cdf, facecolor='blue', interpolate=True, alpha=0.4)
    plt.axvline(l01, linestyle='--', color='k', linewidth=0.8)
    plt.axvline(l99, linestyle='--', color='k', linewidth=0.8)
    plt.ylabel('CDF')
    plt.grid(True, linestyle=':')
    plt.xticks(xticks, xticks_str)
    plt.yticks(yticks, yticks_str)
    # plt.ylim([0., 1.])
    plt.xlim([-10.1, 10.1])
    plt.tight_layout()
    st.pyplot(fig)

    mean, var, skew, kurt = stats.norm.stats(moments='mvsk', loc=param_mean, scale=param_var)
    st.write('The mean of the distributions is also its median and mode (here={0:2.2f}). The variance in this particular example is {1:2.2f}. '
             'Skewness and kurtosis are constant (=0)'.format(mean, var))


    st.subheader('High-dimensional behavior')
    st.write('Suppose we generate a sample of 100 data points from a Gaussian distribution with mean 0 and variance 1. '
             'Using the slider below, you can vary the number of dimensions and we measure the'
             'euclidean distance to the origin. Surprisingly, with increasing dimensionality, the datapoints will be less '
             'likely to lie at the origin and will concentrate around some non-zero distance. Hence, data points stemming from '
             'a high-dimensional Gaussian distribution (with each dimension identical and independent) will approximately lie on '
             'the surface of a hypersphere.')
    param_dims = st.slider('Dimensions', 1, 1000, 1, step=1)
    data_gen = np.random.randn(param_dims, 100)
    # st.write(data_gen.shape)
    distances_origin = np.linalg.norm(data_gen, axis=0)
    # st.write(distances_origin.shape)
    # st.write(np.max(distances_origin) - np.min(distances_origin))
    # bins, _ = np.histogram(distances_origin)

    fig = plt.figure(figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')
    # HISTOGRAM
    _ = plt.hist(distances_origin, bins='auto')
    plt.grid(True, linestyle=':')
    plt.ylabel('Histogram')
    plt.xlim([0, np.max(distances_origin)+4])
    st.pyplot(fig)


@register_distr(name='Laplace')
def dist_laplace():
    st.subheader('Description')
    st.write(wikipedia.summary('Laplace distribution'))
    st.write('The probability density function for laplace is')
    st.latex(r'f(x|\mu,\sigma) = \frac{1}{2\sigma}\exp\left(-\frac{|x - \mu|}{\sigma} \right)')
    st.markdown('for a real number $x$ with support on the whole $\mathbb{R}$.')

    # (A) let the user seletect the parameters
    param_mean = st.slider('Location', -10.0, +10.0, 0.0, step=0.01)
    param_var = st.slider('Scale', 0.01, 10.0, 1.0, step=0.01)

    l01 = stats.laplace.ppf(0.01, loc=param_mean, scale=param_var)
    l99 = stats.laplace.ppf(0.99, loc=param_mean, scale=param_var)

    # (B) generate data based on parameters
    x = np.linspace(-10.0, +10.0, 1000)
    data_pdf = stats.laplace.pdf(x, loc=param_mean, scale=param_var)
    data_cdf = stats.laplace.cdf(x, loc=param_mean, scale=param_var)

    # (C) plot some nice graphics
    xticks = np.zeros(21+1)
    xticks[1:] = np.linspace(-10., +10., 21)
    xticks_str = ['']*xticks.size
    xticks_str[1] = '{0:1.1f}'.format(xticks[1])
    xticks_str[-1] = '{0:1.1f}'.format(xticks[-1])
    xticks_str[0] = '0'

    yticks = np.linspace(0., 1., 10)
    yticks_str = ['']*yticks.size
    yticks_str[0] = '{0:1.1f}'.format(yticks[0])
    yticks_str[-1] = '{0:1.1f}'.format(yticks[-1])

    st.markdown('Following figures show the PDF (red) and CDF (blue) as well as the 1 and the 99 percentile (black dashed lines). ')
    fig = plt.figure(0)
    # PDF
    plt.subplot(2, 1, 1)
    plt.plot(x, data_pdf, '-r')
    plt.fill_between(x, np.zeros(x.size), data_pdf, facecolor='red', interpolate=True, alpha=0.4)
    plt.axvline(l01, linestyle='--', color='k', linewidth=0.8)
    plt.axvline(l99, linestyle='--', color='k', linewidth=0.8)
    plt.grid(True, linestyle=':')
    plt.ylabel('PDF')
    plt.xticks(xticks, '')
    plt.yticks(yticks, yticks_str)
    # plt.ylim([0., 1.])
    plt.xlim([-10.1, 10.1])

    # CDF
    plt.subplot(2, 1, 2)
    plt.plot(x, data_cdf, '-b')
    plt.fill_between(x, np.zeros(x.size), data_cdf, facecolor='blue', interpolate=True, alpha=0.4)
    plt.axvline(l01, linestyle='--', color='k', linewidth=0.8)
    plt.axvline(l99, linestyle='--', color='k', linewidth=0.8)
    plt.ylabel('CDF')
    plt.grid(True, linestyle=':')
    plt.xticks(xticks, xticks_str)
    plt.yticks(yticks, yticks_str)
    # plt.ylim([0., 1.])
    plt.xlim([-10.1, 10.1])

    st.pyplot(fig)

    mean, var, skew, kurt = stats.laplace.stats(moments='mvsk', loc=param_mean, scale=param_var)
    st.write('Mean = median = mode: {0:2.4f}'.format( mean))
    st.write('Variance:  {0:2.4f}'.format(var))
    st.write('Skewness:  {0:2.4f} (is constant, does not depend on the parameters)'.format(skew))
    st.write('Kurtosis:  {0:2.4f} (is constant, does not depend on the parameters)'.format(kurt))


@register_distr(name='Gamma')
def dist_gamma():
    st.subheader('Description')
    st.write(wikipedia.summary('gamma distribution', sentences=8))
    st.write('The probability density function for Gamma is')

    st.latex(r'f(x|a) = \frac{x^{a-1}\exp\left(-x\right)}{\Gamma(a)}')

    st.markdown('for a real number $x \geq 0$, $a > 0$  with support on the whole $\mathbb{R}^+_0$.')

    # (A) let the user seletect the parameters
    param_loc = st.slider('Location', -10.0, +10.0, 0.0, step=0.01)
    param_scale = st.slider('Scale', 0.01, 10.0, 1.0, step=0.01)
    param_a = st.slider('Parameter a', 0.01, 10.0, 1.0, step=0.01)

    l01 = stats.gamma.ppf(0.01, loc=param_loc, scale=param_scale, a=param_a)
    l99 = stats.gamma.ppf(0.99, loc=param_loc, scale=param_scale, a=param_a)

    # (B) generate data based on parameters
    x = np.linspace(-10.0, +10.0, 1000)
    data_pdf = stats.gamma.pdf(x, loc=param_loc, scale=param_scale, a=param_a)
    data_cdf = stats.gamma.cdf(x, loc=param_loc, scale=param_scale, a=param_a)

    # (C) plot some nice graphics
    xticks = np.zeros(21+1)
    xticks[1:] = np.linspace(-10., +10., 21)
    xticks_str = ['']*xticks.size
    xticks_str[1] = '{0:1.1f}'.format(xticks[1])
    xticks_str[-1] = '{0:1.1f}'.format(xticks[-1])
    xticks_str[0] = '0'

    yticks = np.linspace(0., 1., 10)
    yticks_str = ['']*yticks.size
    yticks_str[0] = '{0:1.1f}'.format(yticks[0])
    yticks_str[-1] = '{0:1.1f}'.format(yticks[-1])

    st.markdown('Following figures show the PDF (red) and CDF (blue) as well as the 1 and the 99 percentile (black dashed lines). ')
    fig = plt.figure(0)
    # PDF
    plt.subplot(2, 1, 1)
    plt.plot(x, data_pdf, '-r')
    plt.fill_between(x, np.zeros(x.size), data_pdf, facecolor='red', interpolate=True, alpha=0.4)
    plt.axvline(l01, linestyle='--', color='k', linewidth=0.8)
    plt.axvline(l99, linestyle='--', color='k', linewidth=0.8)
    plt.grid(True, linestyle=':')
    plt.ylabel('PDF')
    plt.xticks(xticks, '')
    plt.yticks(yticks, yticks_str)
    # plt.ylim([0., 1.])
    plt.xlim([-10.1, 10.1])

    # CDF
    plt.subplot(2, 1, 2)
    plt.plot(x, data_cdf, '-b')
    plt.fill_between(x, np.zeros(x.size), data_cdf, facecolor='blue', interpolate=True, alpha=0.4)
    plt.axvline(l01, linestyle='--', color='k', linewidth=0.8)
    plt.axvline(l99, linestyle='--', color='k', linewidth=0.8)
    plt.ylabel('CDF')
    plt.grid(True, linestyle=':')
    plt.xticks(xticks, xticks_str)
    plt.yticks(yticks, yticks_str)
    # plt.ylim([0., 1.])
    plt.xlim([-10.1, 10.1])

    st.pyplot(fig)

    mean, var, skew, kurt = stats.gamma.stats(moments='mvsk', loc=param_loc, scale=param_scale, a=param_a)
    st.write(r'Mean (not equal to median and mode): {0:2.4f}'.format(mean))
    st.write('Variance:  {0:2.4f}'.format(var))
    st.write('Skewness:  {0:2.4f}'.format(skew))
    st.write('Kurtosis:  {0:2.4f}'.format(kurt))


# st.title('Know your distributions')
option = st.selectbox('Choose a distribution', distr_list)
# st.write('You selected:', option)
for i in range(len(distr_list)):
    if distr_list[i] == option:
        distr_func_list[i]()

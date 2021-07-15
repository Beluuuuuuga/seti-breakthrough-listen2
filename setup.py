from setuptools import setup, find_packages

setup(
    name='hedgehog',   
    # version=__version__,
    version='0.0.1',
    description="competition package",
    # long_description="長めの説明",
    # url='必要ならばURL',
    author='Beluga',
    packages=find_packages(),
    # author_email='メールアドレス',
    # license='ライセンス',
    # classifiers=[
        # パッケージのカテゴリー
        # https://pypi.python.org/pypi?:action=list_classifiers
        # から該当しそなもを選んでおけばよい。
    # ],
    # keywords='キーワード',
    install_requires=[
        "pandas>=1.1.1",
    ],
    python_requires=">=3.7.7",
)
# 基于机器学习的 Webshell 发现技术探索

> WebShell 就是以 ASP、PHP、JSP 或者 CGI 等文件形式存在的一种命令执行环境，也可以将其称做为一种网页后门。黑客在入侵了一个网站后，通常会将 ASP 或 PHP 后门文件与网站服务器 web 目录下正常的网页文件混在一起。然后就可以使用浏览器来访问 ASP 或者 PHP 后门，得到一个命令执行环境，以达到控制网站服务器的目的。顾名思义，「web」的含义是显然需要服务器开放 web 服务，「shell」的含义是取得对服务器某种程度上操作权限。WebShell 常常被称为入侵者通过网站端口对网站服务器的某种程度上操作的权限。由于 WebShell 其大多是以动态脚本的形式出现，也有人称之为网站的后门工具。  

  

在攻击链模型中，整个攻击过程分为以下几个步骤：

*   Reconnaissance（踩点）
    
*   Weaponization（组装）
    
*   Delivery（投送）
    
*   Exploitation（攻击）
    
*   Installation（植入）
    
*   C2（控制）
    
*   Actions （行动）
    

![](https://mmbiz.qpic.cn/mmbiz_png/0vU1ia3htaaMx2E2hxNqRvBrO3yBLYtZNrwkbzENcmM4zGtSnCnzOEObNJ2RzFw96IOOkqDVxNPibRd6emf81XbQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

攻击链模型

在针对网站的攻击中，通常是利用上传漏洞，上传 WebShell，然后通过 WebShell 进一步控制web服务器，对应攻击链模型是 Install 和 C2环节。

常见的 WebShell 检测方法主要有以下几种：

静态检测，通过匹配特征码，特征值，危险函数函数来查找 WebShell 的方法，只能查找已知的 WebShell，并且误报率漏报率会比较高，但是如果规则完善，可以减低误报率，但是漏报率必定会有所提高。

动态检测，执行时刻表现出来的特征，比如数据库操作、敏感文件读取等。

语法检测，根据 PHP 语言扫描编译的实现方式，进行剥离代码、注释，分析变量、函数、字符串、语言结构的分析方式，来实现关键危险函数的捕捉方式。这样可以完美解决漏报的情况。但误报上，仍存在问题。

统计学检测，通过信息熵、最长单词、重合指数、压缩比等检测。

本章主要以常见的WebShell数据集为例子介绍基于WebShell文件特征的检测技术。 介绍WebShell检测使用的数据集以及对应的特征提取方法，介绍使用的模型以及对应的验证结果，包括朴素贝叶斯和深度学习的MLP、CNN。基于WebShell文件访问特征的检测方法不在本章范围内。

#### 数据集

数据集包含 WebShell 样本2616个，开源软件PHP文件9035个。

WebShell 数据来自互联网上常见的 WebShell 样本，数据来源来自 github 上相关项目，为了演示方便，全部使用了基于 PHP 的 WebShell 样本。

![](https://mmbiz.qpic.cn/mmbiz_png/0vU1ia3htaaMx2E2hxNqRvBrO3yBLYtZNjAIJibxHgp4M6touNB0bgVf5N9Vca5DcnHiaQudFas4H7Ik9mWq4QwMw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

github 上 WebShell 相关项目

白样本主要使用常见的基于 PHP 的开源软件，主要包括以下几种。

**WordPress**

WordPress 是一种使用 PHP 语言开发的博客平台，用户可以在支持 PHP 和 MySQL 数据库的服务器上架设属于自己的网站。也可以把 WordPress 当作一个内容管理系统（CMS）来使用。

WordPress 是一款个人博客系统，并逐步演化成一款内容管理系统软件，它是使用 PHP 语言和 MySQL 数据库开发的。用户可以在支持 PHP 和 MySQL 数据库的服务器上使用自己的博客。

WordPress 有许多第三方开发的免费模板，安装方式简单易用。不过要做一个自己的模板，则需要你有一定的专业知识。

比如你至少要懂的标准通用标记语言下的一个应用 HTML 代码、CSS、PHP 等相关知识。

WordPress 官方支持中文版，同时有爱好者开发的第三方中文语言包，如 wopus 中文语言包。

WordPress 拥有成千上万个各式插件和不计其数的主题模板样式。

项目地址为：https://wordpress.org/

![](https://mmbiz.qpic.cn/mmbiz_png/0vU1ia3htaaMx2E2hxNqRvBrO3yBLYtZN1W7HAtYlPRfWqK7BfDwd7zxMqEtQO6vK0rw8AWyJtMqMXEw45OR4Nw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

WordPress 主页

**PHPCMS**

PHPCMS 是一款网站管理软件。该软件采用模块化开发，支持多种分类方式，使用它可方便实现个性化网站的设计、开发与维护。

它支持众多的程序组合，可轻松实现网站平台迁移，并可广泛满足各种规模的网站需求，可靠性高，是一款具备文章、下载、图片、分类信息、影视、商城、采集、财务等众多功能的强大、易用、可扩展的优秀网站管理软件。

PHPCMS 由国内80后知名创业者钟胜辉（网名：淡淡风）于2005年创办，是国内知名的站长建站工具。

2009年，PHPCMS 创办人钟胜辉离开 PHPCMS，创办国内针对媒体领域的 CMS 产品 CmsTop（思拓合众）。

项目地址为：http://www.phpcms.cn/

![](https://mmbiz.qpic.cn/mmbiz_png/0vU1ia3htaaMx2E2hxNqRvBrO3yBLYtZNXZuczIpJS4zxx5ACBnr1oPgULzQIOHmIISBNj3zic53Bm7iaury5XIYg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

phpcms主页

**phpMyAdmin**

phpMyAdmin 是一个以PHP为基础，以 Web-Base 方式架构在网站主机上的 MySQL 的数据库管理工具，让管理者可用 Web 接口管理 MySQL 数据库。

借由此 Web 接口可以成为一个简易方式输入繁杂 SQL 语法的较佳途径，尤其要处理大量资料的汇入及汇出更为方便。

其中一个更大的优势在于由于 phpMyAdmin 跟其他 PHP 程式一样在网页服务器上执行，但是您可以在任何地方使用这些程式产生的 HTML 页面，也就是于远端管理 MySQL 数据库，方便的建立、修改、删除数据库及资料表。

也可借由 phpMyAdmin 建立常用的 php 语法，方便编写网页时所需要的 sql 语法正确性。

项目地址为：https://www.phpMyAdmin.net/

![](https://mmbiz.qpic.cn/mmbiz_png/0vU1ia3htaaMx2E2hxNqRvBrO3yBLYtZNjxXbbWBmLNu9PhD67LQYomaFPLv8icAAiaMR67773sKt1U1lbpr6yYibQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

phpMyAdmin 主页

**Smarty**

Smarty 是一个使用 PHP 写出来的模板引擎，是目前业界最著名的PHP模板引擎之一。

它分离了逻辑代码和外在的内容，提供了一种易于管理和使用的方法，用来将原本与 HTML 代码混杂在一起 PHP 代码逻辑分离。

简单的讲，目的就是要使 PHP 程序员同前端人员分离，使程序员改变程序的逻辑内容不会影响到前端人员的页面设计，前端人员重新修改页面不会影响到程序的程序逻辑，这在多人合作的项目中显的尤为重要。

项目地址为：https://github.com/smarty-php/smarty

**Yii**

Yii 是一个基于组件的高性能 PHP 框架，用于开发大型 Web 应用。Yii 采用严格的 OOP 编写，并有着完善的库引用以及全面的教程。从 MVC，DAO/ActiveRecord，widgets，caching，等级式 RBAC，Web 服务，到主题化，I18N和L10N，Yii 提供了今日 Web 2.0应用开发所需要的几乎一切功能。事实上，Yii是最有效率的 PHP 框架之一。

Yii 是一个高性能的 PHP5 的 web 应用程序开发框架。通过一个简单的命令行工具 yiic 可以快速创建一个 web 应用程序的代码框架，开发者可以在生成的代码框架基础上添加业务逻辑，以快速完成应用程序的开发。

项目地址为：http://www.yiiframework.com/

![](https://mmbiz.qpic.cn/mmbiz_png/0vU1ia3htaaMx2E2hxNqRvBrO3yBLYtZNqicibEzSliaVpDGRrEyGZJCKE3vB3m2VrFiaWl8GZgjd7CSrROoX0J5mVw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

Yii主页

#### 特征提取

**方法一：词袋 &TF-IDF 模型**

我们使用最常见的词袋模型 &TF-IDF 提取文件特征。

把一个 PHP 文件作为一个完整的字符串处理，定义函数 load\_one\_file 加载文件到一个字符串变量中返回。

```
def load_one_file(filename):
    x=""
    with open(filename) as f:
        for line in f:
            line = line.strip('')
            x+=line
    return x
```

由于开源软件中包含大量图片、js 等文件，所以遍历目录时需要排除非php文件。

另外开源软件的目录结构相对复杂，不像前面章节的垃圾邮件、垃圾短信等是平面目录结构，所以要求我们递归访问指定目录并加载指定文件。

```
def load_files_re(dir):
    files_list = []
    g = os.walk(dir)
    for path, d, filelist in g:
        for filename in filelist:
            if filename.endswith('.php'):
                fulepath =os.path.join(path, filename)
                print "Load %s"% fulepath
                t = load_file(fulepath)
                files_list.append(t)
    return files_list
```

加载搜集到的 WebShell 样本，并统计样本个数，将 WebShell 样本标记为1。

```
WebShell_files_list = load_files_re(WebShell_dir)y1=[1]*len(WebShell_files_list)black_count=len(WebShell_files_list)
```

加载搜集到的开源软件样本，并统计样本个数，将开源软件样本标记为0。

```
wp_files_list =load_files_re(whitefile_dir)y2=[0]*len(wp_files_list)white_count=len(wp_files_list)
```


将 WebShell 样本和开源软件样本合并。

```
x=WebShell_files_list+wp_files_listy=y1+y2
```

使用 2-gram 提取词袋模型，并使用TF-IDF进行处理。

```
CV = CountVectorizer(ngram_range=(2, 2),decode_error="ignore",max_features=max_features,token_pattern = r'w+',min_df=1, max_df=1.0)x=CV.fit_transform(x).toarray()transformer = TfidfTransformer(smooth_idf=False)x_tfidf = transformer.fit_transform(x)x = x_tfidf.toarray()
```

所谓的2-gram是词袋模型的一个细分类别，也有的机器学习书籍里面单独把2-gram或者说n-gram作为单独的模型介绍。n-gram基于这样一种假设，第n个单词只和它前面的n-1个词有关联，每n个单词作为一个处理单元。

![](https://mmbiz.qpic.cn/mmbiz_png/0vU1ia3htaaMx2E2hxNqRvBrO3yBLYtZNyshyuE8v6YFJ4Dy769bXG45I3VsZY2QKzwhSqCaIZ2Zib8KDbc8P1Ug/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

2-gram举例

通过设置`CountVectorizer`函数的`ngram_range`参数和`token_pattern`即可实现n-gram，其中`ngram_range`表明n-gram的n取值范围，如果是2-gram设置成（2，2）即可。`token_pattern`表明词切分的规则，通常设置为`r'w+'`即可。

划分训练集与测试集，测试集的比例为40%。

```
x_train, x_test, y_train,y_test = train_test_split(x, y, test_size=0.4, random_state=0)
```

**方法二：opcode&n-gram模型**

opcode 是计算机指令中的一部分，用于指定要执行的操作， 指令的格式和规范由处理器的指令规范指定。

除了指令本身以外通常还有指令所需要的操作数，可能有的指令不需要显式的操作数。这些操作数可能是寄存器中的值，堆栈中的值，某块内存的值或者IO端口中的值等等。

通常 opcode 还有另一种称谓：字节码(byte codes)。 例如Java虚拟机(JVM)，.NET 的通用中间语言`(CIL: Common Intermeditate Language)`等等。

PHP 中的 opcode 则属于前面介绍中的后着，PHP 是构建在 Zend 虚拟机(Zend VM)之上的。

PHP 的 opcode 就是 Zend 虚拟机中的指令，常见的 opcode 如下图所示。

![](https://mmbiz.qpic.cn/mmbiz_png/0vU1ia3htaaMx2E2hxNqRvBrO3yBLYtZNsevogd17ceDTZf5YKqAcKrib0uq6w1LLM7PN5OVTEytAJIrfic0rUksA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

PHP常见 opcode

通常可以通过 PHP 的 VLD（Vulcan Logic Dumper，逻辑代码展现)是扩展来查看 PHP 文件对应的 opcode。

wget http://pecl.php.net/get/vld-0.13.0.tgztar zxvf vld-0.13.0.tgzcd ./vld-0.13.0/configure--with-php-config=/usr/local/php/bin/php-config --enable-vldmake && makeinstall

然后在php.ini配置文件中添加`extension=vld.so` 用于激活VLD，其中`php.ini`默认位置位于lib目录中。VLD还可以从github上下载并安装，步骤为：

```
git clone https://github.com/derickr/vld.gitcd vldphpize./configuremake && makeinstall
```

VLD项目的主页为：

http://pecl.php.net/package/vld

![](https://mmbiz.qpic.cn/mmbiz_png/0vU1ia3htaaMx2E2hxNqRvBrO3yBLYtZNldvHCIpKqfLicvKvVJzPDwHMJj1ibI8vm2DFBHo951rZm2gOcnwOticVg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

VLD 扩展下载主页

以 PHP 文件`hello.php`为例：

```
 <?php   echo"Hello World";   $a = 1 +1;   echo $a;?>
```

通过使用 PHP 的 VLD 扩展查看对应的 opcode，其中 vld.active=1 表示激活 VlD，`vld.execute=0`表示只解析不执行。

```
php -dvld.active=1 -dvld.execute=0hello.php
```

显示结果为：

```
function name:  (null)number of ops:  5compiled vars:  !0 = $aline     #* E I O op                         fetch          ext  return  operands-----------------------------------------------------------------------------   2     0  E >  ECHO                                              'Hello+World'   3     1       ADD                                            ~0      1, 1         2       ASSIGN                                                 !0, ~0   4     3       ECHO                                                   !0   6     4     > RETURN                                                 1branch: #  0; line:    2-    6; sop:     0; eop:     4; out1:  -2path #1: 0,
```

对应的 opcode 为：

```
ECHO     ADD       ASSIGNECHO
```

以一个常见的一句话木马为例：

```
<?php         echo $_GET['r'];?>
```

通过VLD查看的结果为：

```
function name:  (null)number of ops:  5compiled vars:  noneline     #* E I O op                         fetch          ext  return  operands-------------------------------------------------------------------------------------   2     0  E >  FETCH_R                     global              $0     '_GET'         1       FETCH_DIM_R                                    $1      $0, 'r'         2       ECHO                                                   $1   4     3       ECHO                                                   '+%0A'         4     > RETURN                                                 1branch: #  0; line:    2-    4; sop:     0; eop:     4; out1:  -2path #1: 0,
```

对应的 opcode 为：

```
FETCH_RFETCH_DIM_R ECHO ECHORETURN
```

使用2-gram对 opcode 进行分组，结果为：

```
(FETCH_R, FETCH_DIM_R) (FETCH_DIM_R, ECHO) (ECHO, ECHO) (ECHO, RETURN)
```

完整的处理流程为：

![](https://mmbiz.qpic.cn/mmbiz_png/0vU1ia3htaaMx2E2hxNqRvBrO3yBLYtZNaDe3bZjiaTqZKN6HaIkibsCHDicKwdqZZ3obhIBnBy4nucuuqArwvtkCQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

PHP 代码处理流程图

代码实现方面，首先使用 VLD 处理 PHP 文件，把处理的结果保存在字符串中。

```
t=""cmd=php_bin+" -dvld.active=1 -dvld.execute=0 "+file_pathoutput=commands.getoutput(cmd)
```

PHP 的 opcode 都是由大写字母和下划线组成的单词，使用findall函数从字符串中提取全部满足条件的 opcode，并以空格连接成一个新字符串。

```
t=outputtokens=re.findall(r's([A-Z_]+)s',output)t=" ".join(tokens)
```

遍历读取指定目录下全部 PHP 文件，保存其对应的 opcode 字符串。

```
defload_files_opcode_re(dir):    files_list = []    g = os.walk(dir)    for path, d, filelist in g:        for filename in filelist:            if filename.endswith('.php'):                fulepath =os.path.join(path, filename)                print "Load %sopcode" % fulepath                t =load_file_opcode(fulepath)                files_list.append(t)    return files_list
```

依次读取保存 WebShell 样本以及正常PHP文件的目录，加载对应的opcode字符串，其中标记 WebShell 为1，正常PHP文件为0。

```
WebShell_files_list= load_files_re(WebShell_dir)y1=[1]*len(WebShell_files_list)black_count=len(WebShell_files_list)wp_files_list =load_files_re(whitefile_dir)y2=[0]*len(wp_files_list)white_count=len(wp_files_list)
```

使用2-gram处理opcode字符串,其中通过设置`ngram_range=(2, 2)`就可以达到使用2-gram的目的，同理如果使用3-gram设置`ngram_range=(3, 3)`即可。

```
CV= CountVectorizer(ngram_range=(2, 2), decode_error="ignore",max_features=max_features,token_pattern = r'w+',min_df=1, max_df=1.0)x=CV.fit_transform(x).toarray()
```

使用TF-IDF进一步处理。

```
transformer= TfidfTransformer(smooth_idf=False)x_tfidf = transformer.fit_transform(x)x = x_tfidf.toarray()
```

另外，开发调试阶段会频繁解析相同的 PHP 文件获取对应的 opcode，可以使用 PHP 的 opcode 缓存技术提高效率。

opcode 缓存技术可以有效减少不必要的编译步骤，减少cpu和内存的消耗。正常情况下 PHP 代码的执行过程会经历文本扫描、语法解析、创建 opcode、执行 opcode 这几部。

![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

未使用 opcode 缓存的情况下 PHP 代码执行过程

使用了 opcode 缓存技术后，对于曾经解析过的 PHP 文件，opcode 会缓存下来，遇到同样内容的PHP文件就可以直接进入 opcode 执行阶段。

![](https://mmbiz.qpic.cn/mmbiz_png/0vU1ia3htaaMx2E2hxNqRvBrO3yBLYtZNkRiblP7gmQWmIGsQBFicj1YXxr00wjcicMm1e4yz2B3yThiaPeGBb5B5yw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

使用 opcode 缓存的情况下 PHP 代码执行过程.

开启 opcode 的缓存功能非常方便，PHP 5.5.0 以后在编译 PHP 源码的时候开启— enable-opcache，编译选型为：

```
./configure--prefix=/opt/php --enable-opcacheconfig.status:creating php5.specconfig.status:creating main/build-defs.hconfig.status:creating scripts/phpizeconfig.status:creating scripts/man1/phpize.1config.status:creating scripts/php-configconfig.status:creating scripts/man1/php-config.1config.status:creating sapi/cli/php.1config.status:creating sapi/cgi/php-cgi.1config.status:creating ext/phar/phar.1config.status:creating ext/phar/phar.phar.1config.status:creating main/php_config.hconfig.status:executing default commands
```

编译安装

```
make-j4 & make install
```

修改配置文件 php.ini，加载对应的动态库。

```
zend_extension=/full/path/to/opcache.so
```

配置opcode缓存对应的配置选项,典型的配置内容如下所示。

```
engine= Onzend_extension=/lib/php/extensions/no-debug-non-zts-20131226/opcache.soopcache.memory_consumption=128opcache.interned_strings_buffer=8opcache.max_accelerated_files=4000opcache.revalidate_freq=60opcache.fast_shutdown=1opcache.enable_cli=1opcache.enable=1
```

**方法三：opcode 调用序列模型**

在 opcode&n-gram 模型中，我们假设第n个 opcode 之与前n-1个 opcode 有关联，现在我们以一个更加长的时间范围来看 opcode 的调用序列，把整个 PHP 的 opcode 当成一个调用序列来分析。

为了便于程序处理，截取整个文件 opcode 的固定长度的 opcode 序列分析，超过固定长度的截断，不足的使用0补齐。以一个常见的一句话木马为例：

```
<?php         echo $_GET['r'];?>
```

![](https://mmbiz.qpic.cn/mmbiz_png/0vU1ia3htaaMx2E2hxNqRvBrO3yBLYtZN5a2TOGdQfKFp52Ls1o28bFJrOqESNsKStbicyA4jXHUWK5LVT9Y9M3A/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

图11-13     解析 PHP 文件获取 opcode 调用序列的过程

该文件通过 VLD 处理获得对应的 opcode 为：

```
FETCH_RFETCH_DIM_R ECHO ECHORETURN
```

获得对应的 opcode 序列为：

```
（FETCH_R，FETCH_DIM_R，ECHO，ECHO，RETURN ）
```

#### 模型训练与验证

**方法一：朴素贝叶斯算法**

使用朴素贝叶斯算法，特征提取使用词袋&TF-IDF模型，完整的处理流程为：

1.  将 WebShell 样本以及常见 PHP 开源软件的文件提取词袋。
    
2.  使用 TF-IDF 处理。
    
3.  随机划分为训练集和测试集。
    
4.  使用朴素贝叶斯算法在训练集上训练，获得模型数据。
    
5.  使用模型数据在测试集上进行预测。
    
6.  验证朴素贝叶斯算法预测效果。
    

![](https://mmbiz.qpic.cn/mmbiz_png/0vU1ia3htaaMx2E2hxNqRvBrO3yBLYtZNOxuLfgQ7kb75bxqibficUqib4nh3N6q23aHLpAfwN8NbCrWiafYmNFTBvQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

  

特征提取使用词袋&TF-IDF模型算法使用朴素贝叶斯的流程图

实例化朴素贝叶斯算法，并在训练集上训练数据，针对测试集进行预测。

```
gnb = GaussianNB()gnb.fit(x_train,y_train)y_pred=gnb.predict(x_test)
```

评估结果的准确度和TP、FP、TN、FN四个值。

```
print metrics.accuracy_score(y_test, y_pred)print metrics.confusion_matrix(y_test, y_pred)
```

在词袋最大特征数为15000的情况下，使用词袋&TF-IDF模型时，TP、FP、TN、FN矩阵如下表所示。

表1-1    基于词袋&TF-IDF模型的朴素贝叶斯验证结果

| 类型名称 | 相关 | 不相关 |
| --- | --- |--- |
| 检索到 | 3566 | 52 |
| 未检索到 | 71 | 972 |




整个系统的准确率为94.92%，召回率为93.19%。

完整输出结果为：

```
metrics.accuracy_score:0.97361081313metrics.confusion_matrix:[[3566   52] [  71  972]]metrics.precision_score:0.94921875metrics.recall_score:0.931927133269metrics.f1_score:0.940493468795
```

**方法二：深度学习算法之MLP**

使用MLP算法，隐含层设计为2层，每次节点数分别为5和2。

![](https://mmbiz.qpic.cn/mmbiz_png/0vU1ia3htaaMx2E2hxNqRvBrO3yBLYtZNjNpPRibZ5qXvjJV74OnTTib5koM3TibvFtTtW7EHg2EYCMicnTu48xQuPg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

MLP隐藏层设计  

使用MLP算法，特征提取使用词袋&TF-IDF模型，完整的处理流程为：

1.  将WebShell样本以及常见PHP开源软件的文件提取词袋。
    
2.  使用TF-IDF处理。
    
3.  随机划分为训练集和测试集。
    
4.  使用MLP算法在训练集上训练，获得模型数据。
    
5.  使用模型数据在测试集上进行预测。
    
6.  验证MLP算法预测效果。
    

实例化MLP算法，并在训练集上训练数据，针对测试集进行预测。

```
clf = MLPClassifier(solver='lbfgs',                    alpha=1e-5,                    hidden_layer_sizes =(5, 2),                    random_state = 1)print  clfclf.fit(x_train, y_train)y_pred = clf.predict(x_test)
```

评估结果的 TP、FP、TN、FN 四个值。

```
print metrics.accuracy_score(y_test, y_pred)print metrics.confusion_matrix(y_test, y_pred)
```

评估结果的准确率与召回率以及F1分值。

```
print "metrics.precision_score:"print metrics.precision_score(y_test, y_pred)print "metrics.recall_score:"print metrics.recall_score(y_test, y_pred)print "metrics.f1_score:"print metrics.f1_score(y_test,y_pred)
```

在词袋最大特征数为15000且同时使用TF-IDF模型的情况下， TP、FP、TN、FN矩阵如下表所示。

表1-2    基于词袋和 TF-IDF 模型的MLP验证结果

| 类型名称 | 相关 | 不相关 |
| --- | --- |--- |
| 检索到 | 3583 | 35 |
| 未检索到 | 51 | 992 |










准确率为96.59%，召回率为95.11%。

完整输出结果为：

```
metrics.confusion_matrix:[[3583   35] [  51  992]]metrics.precision_score:0.965920155794metrics.recall_score:0.951102588686metrics.f1_score:0.95845410628
```

使用MLP算法，特征提取使用特征提取使用`opcode&n-gram`,完整的处理流程为：

1.  将 WebShell 样本以及常见 PHP 开源软件的文件提取 opcode.
    
2.  使用 n-gram 处理。
    
3.  随机划分为训练集和测试集。
    
4.  使用 MLP 算法在训练集上训练，获得模型数据。
    
5.  使用模型数据在测试集上进行预测。
    
6.  验证 MLP 算法预测效果。
    

特征提取使用`opcode&n-gram,n`取4，最大特征数取2000的情况下，TP、FP、TN、FN矩阵如下表所示。

表1-3    基于opcode&n-gram模型的MLP验证结果


| 类型名称 | 相关 | 不相关 |
| --- | --- |--- |
| 检索到 | 2601 | 97 |
| 未检索到 | 20 | 484 |












准确率为83.30%，召回率为96.03%。

完整输出结果为：

```
0.963460337289metrics.confusion_matrix:[[2601   97] [  20 484]]metrics.precision_score:0.833046471601metrics.recall_score:0.960317460317metrics.f1_score:0.892165898618
```

**方法三：深度学习算法之CNN**

使用方法二中生成的 opcode&n-gram 数据，算法使用CNN，完整的处理流程为：

1.  将 WebShell 样本以及常见 PHP 开源软件的文件提取 opcode.
    
2.  使用 n-gram 处理。
    
3.  随机划分为训练集和测试集。
    
4.  使用 CNN 算法在训练集上训练，获得模型数据。
    
5.  使用模型数据在测试集上进行预测。
    
6.  验证 CNN 算法预测效果。
    

使用方法二中生成的 opcode&n-gram 数据，获得训练数据集和测试数据集。

```
x, y = get_feature_by_opcode()x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.4, random_state = 0)
```

将训练和测试数据进行填充和转换，不到最大长度的数据填充0，由于是二分类问题，把标记数据二值化。定义输入参数的最大长度为文档的最大长度。

```
trainX = pad_sequences(trainX, maxlen=max_document_length,value=0.)testX = pad_sequences(testX, maxlen=max_document_length, value=0.)# Converting labels to binary vectorstrainY = to_categorical(trainY, nb_classes=2)testY = to_categorical(testY, nb_classes=2)network = input_data(shape=[None,max_document_length],name='input')
```

定义CNN模型，使用3个数量为128，长度分别为3、4、5的一维卷积函数处理数据。

```
network = tflearn.embedding(network, input_dim=1000000,output_dim=128)branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu',regularizer="L2")branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu',regularizer="L2")branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu',regularizer="L2")network = merge([branch1, branch2, branch3], mode='concat', axis=1)network = tf.expand_dims(network, 2)network = global_max_pool(network)network = dropout(network, 0.8)network = fully_connected(network, 2, activation='softmax')network = regression(network, optimizer='adam', learning_rate=0.001,loss='categorical_crossentropy', name='target')
```

实例化CNN对象并进行训练数据，一共训练5轮。

```
model = tflearn.DNN(network, tensorboard_verbose=0)model.fit(trainX, trainY,          n_epoch=5, shuffle=True,validation_set=0.1,          show_metric=True,batch_size=100,run_id="webshell")
```

完整的 CNN 结构如下图所示。

![](https://mmbiz.qpic.cn/mmbiz_png/0vU1ia3htaaMx2E2hxNqRvBrO3yBLYtZNibAUcWyyltTBibbCIPBBbtCCGXapSCKIP6uiaicdf48pwBGkN1d8YSkY5g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

用于识别 WebShell 的CNN结构图

考核 CNN 对应的准确率、召回率，误报数和漏报数。

```
print "metrics.accuracy_score:"print metrics.accuracy_score(y_test, y_pred)print "metrics.confusion_matrix:"print metrics.confusion_matrix(y_test, y_pred)print "metrics.precision_score:"print metrics.precision_score(y_test, y_pred)print "metrics.recall_score:"print metrics.recall_score(y_test, y_pred)print "metrics.f1_score:"print metrics.f1_score(y_test,y_pred)
```

运行程序，经过5轮训练，在的情况下，使用`opcode&n-gram`模型时，n取4，TP、FP、TN、FN矩阵如下表所示。

表1-4    基于opcode&n-gram模型的CNN验证结果

| 类型名称 | 相关 | 不相关 |
| --- | --- |--- |
| 检索到 | 2669 | 29 |
| 未检索到 | 367 | 137 |





整个系统的准确率为82.53%，召回率为27.18%。

完整输出结果为：

```
metrics.accuracy_score:0.87632729544metrics.confusion_matrix:[[2669   29] [ 367 137]]metrics.precision_score:0.825301204819metrics.recall_score:0.271825396825metrics.f1_score:0.408955223881
```

使用方法三中生成的 opcode 序列数据，算法使用CNN，完整的处理流程为：

1.  将 WebShell 样本以及常见 PHP 开源软件的文件提取 opcode.
    
2.  使用词袋处理，针对 opcode 进行编号，生成 opcode 序列。
    
3.  随机划分为训练集和测试集。
    
4.  使用 CNN 算法在训练集上训练，获得模型数据。
    
5.  使用模型数据在测试集上进行预测。
    
6.  验证 CNN 算法预测效果。
    

使用方法三中 opcode 调用序列编码后的数据，获得训练数据集和测试数据集。

```
x_train, x_test, y_train, y_test= get_feature_by_opcode ()
```

运行程序，经过5轮训练，在 opcode 序列长度为3000的情况下，使用 opcode 序列模型时，TP、FP、TN、FN矩阵如下表所示。

表1-5    基于 opcode 序列模型的 CNN 验证结果

| 类型名称 | 相关 | 不相关 |
| --- | --- |--- |
| 检索到 | 2685 | 13 |
| 未检索到 | 89 | 415 |

整个系统的准确率为96.96%，召回率为82.34%。

完整输出结果为：

```
metrics.accuracy_score:0.968144909432metrics.confusion_matrix:[[2685   13] [  89 415]]metrics.precision_score:0.969626168224metrics.recall_score:0.823412698413metrics.f1_score:0.890557939914
```

#### 本章小结

本章基于搜集的 PHP 的 WebShell 数据集介绍了 WebShell 的识别方法。针对 PHP 的 WebShell 数据集，特征提取方法有词袋`&TF-IDF`、`opcode&n-gram`以及`opcode`序列三种方法。训练模型介绍了朴素贝叶斯以及深度学习的 MLP 和 CNN 算法，其中基于基于词袋和 TF-IDF 模型的 MLP 准确率和召回率综合表现最佳，基于 opcode 序列模型的 CNN 准确率较高。

本文部分截取自我书中的内容，有兴趣的读者也可以在京东或者亚马逊搜索我的AI三部曲<web安全之机器学习入门>和<web安全之深度学习实战>

本文转载自我在GitChat的一次在线分享,原文链接请点击 "阅读原文"

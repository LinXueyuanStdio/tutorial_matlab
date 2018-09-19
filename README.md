# tutorial_matlab
matlab

### 0. 认识MATLAB

**MATLAB**是一种语法简单用途广泛的编程语言，既可以用于编写脚本，函数，也可以用于面向对象的程序开发或开发GUI界面。**MATLAB**被广泛应用于数值计算，图像处理，机器学习等领域。

### 1. 变量与矩阵

**MATLAB**在变量声明是不需要指出变量的类型。

~~~matlab
clear; %清空内存
clc; %清空命令行
r1=1; %为一个变量赋值
z1=1+sqrt(3)*i; %赋值一个复数 sqrt()开方运算
z_real=real(z1); %复数的实部
z_img=imag(z1); %复数的虚部
z_abs=abs(z1); %复数的模
z_ang=angle(z1); %复数的幅角
z2=z1^2; %平方运算
~~~

**MATLAB**的数组索引从1开始，这点需要牢记。

~~~matlab
arr1=rand(1,5); %arr1=[0.1418,0.4217,0.9157,0.7922,0.9594]
arr2=zeros(1,5); %arr2=[0,0,0,0,0]
arr3=ones(1,5); %arr3=[1,1,1,1,1]
arr4=linspace(1,2,5); %arr4=[1,1.25,1.5,1.75,2]
mat1=rand(3,3); %随机生成3*3矩阵
mat2=[1,2,3;4,5,6;7,8,9];
~~~

### 2. 分支与循环

**MATLAB**常用的分支语句有__if-else__和__switch-case__，以下来自**MATLAB**帮助文档关于__if__的介绍。(__doc if__)

```matlab
limit = 0.75;
A = rand(10,1)
if any(A > limit)
    disp('There is at least one value above the limit.')
else
    disp('All values are below the limit.')
end
```

**MATLAB**常用的循环有__while__循环和__for__循环，以下来自**MATLAB**帮助文档关于__for__的介绍。(__doc for__)

```matlab
for v = 1.0:-0.2:0.0
   disp(v)
end

for v = [1 5 8 17]
   disp(v)
end
```

### 3. 函数及函数句柄

这里分别使用`函数`和`函数句柄`的方法来生成__Fibonacci__数列。

需要注意函数名和文件名要保持一致，以下先使用`函数`的方式：

~~~matlab
function y = fibonacci (x)
if x == 1 || x==2
    y = 1;
    return % return可以不写
else 
    y = fibonacci(x-1) + fibonacci(x-2);
    return
end
~~~

以下是使用`函数句柄`的方式：

~~~matlab
fibo=@(n) (((1+sqrt(5))/2)^n-((1-sqrt(5))/2)^n)/sqrt(5);
fn=zeros(1,100);
for i=1:1:100
    fn(i)=fibo(i);
end
~~~

### 4. 绘图函数

**MATLAB**中有很多绘图函数。以下只演示常用的

使用`ezplot`:

~~~matlab
figure ('name','ezplot_demo');
hold on;
ezplot('(1-y)*(1-x)-0.05*y',[0,1],[0,1]);
ezplot('(1-x)*(1-y)-(0.15+0.6*y)*x',[0,1],[0,1]);
legend('(1-y)*(1-x)-0.05*y','(1-x)*(1-y)-(0.15+0.6*y)*x');
hold off;
title('ezplot demo');
xlabel( 'x' );ylabel( 'y' );
~~~

使用`plot`绘制曲线：

~~~matlab
figure ('name','normal dist');
x=-10:0.001:10;
p=normpdf(x,0,1);
plot(x,p);
title('plot demo normal dist');
xlabel('x');ylabel('p')
~~~

使用`surf`绘制三维曲面：

~~~matlab
[x,y]=meshgrid(0:0.05:1,0:0.05:1); %生成网格数据
v=exp(y).*sin(x);
figure ('name','surf_demo');
surf(x,y,v);
title('surf demo');
xlabel('x');ylabel('y');zlabel('v');
~~~

### 5. 数值微积分

使用dx=0.000001为步长的向前差分求sin(x)的导数：

$$f^,(x)=\frac{f(x+dx)-f(x)}{dx}$$

~~~matlab
figure ('name','diff demo1');
x=linspace(0,10,100);
y=sin(x);
dx=0.000001;dydx=[];
for i=1:100
    dydx(i)=(sin(x(i)+dx)-y(i))/dx;
end
plot(x,y,'r',x,dydx,'b');
legend('sin(x)','cos(x)');
title('diff demo');
xlabel('x');ylabel('y')
~~~

使用**MATLAB**的差分工具`diff`计算导数，以下来自**MATLAB**帮助文档关于__diff__的介绍。(__doc diff__)：

~~~matlab
h = 0.001;       % step size
X = -pi:h:pi;    % domain
f = sin(X);      % range
Y = diff(f)/h;   % first derivative
Z = diff(Y)/h;   % second derivative
plot(X(:,1:length(Y)),Y,'r',X,f,'b', X(:,1:length(Z)),Z,'k')
~~~

使用矩形法计算$\int_0^1x^2dx$：

$$\int_a^bf(x)dx=\frac{b-a}{n}\sum_{i=1}^nf(x_i)$$

~~~matlab
n=100000;a=0;b=1; %取步长为100000
x=a:1/n:b;
dx=(b-a)/n;x=x+dx/2;
s=x.^2; %采样
int=dx*sum(s);
~~~

调用**MATLAB**中的`quad`函数使用__Simpson__法计算数值积分：

~~~matlab
func=@(x)x.^2;
int=quad(func,0,1)
~~~

### 6. 常微分方程（组）的数值解

使用__Euler__法计算常微分方程（误差较大，不推荐）：

$$\frac{dy}{dx}=x^2+y^2+3x-2y$$

$$y|_{x=0}=1$$

取时间步长为h，则

$$y(x_{n+1})=y(x_n)+f(y(x_n),x_n)*h$$

~~~matlab
function matlab_demo
    func=@(x,y)x.^2+y.^2+3*x-2*y
    [x,y]=euler(func,[0,1],1,0.01)
    plot(x,y)
return

function [x,y]=euler(fun,xspan,y0,h)
    x=xspan(1):h:xspan(2)
    y(1)=y0;
    for n=1:length(x)-1
        y(n+1)=y(n)+h*feval(fun,x(n),y(n))
    end
return
~~~

使用45阶__Runge-Kutta__算法`ode45`计算常微分方程组：

$$\frac{dx}{dt}=2x-3y$$

$$\frac{dy}{dt}=x+2y$$

$$x|_{t=0}=1$$

$$y|_{t=0}=1$$

~~~matlab
function ode_demo
y0=[1,1];
tspan=0:0.01:5;
option = odeset('AbsTol',1e-4);
[t,x]=ode45(@dfunc,tspan,y0,option);
figure('name','ode45 demo');
plot(t,x(:,1),'r',t,x(:,2),'b');
return

function dx=dfunc(t,x)
dx=zeros(2,1);
dx(1)=2*x(1)-3*x(2); % x(1)=x
dx(2)=x(1)+2*x(2); % x(2)=y
return
~~~

### 7. 偏微分方程（组）的数值解

使用`pdepe`进行微分方程（组）的求解，需要先将微分方程（组），以及边界和初值条件化为如下形式：

$$c(x,t,\frac{\partial{u}}{\partial{x}})\frac{\partial{u}}{\partial{t}}=x^{-m}\frac{\partial}{\partial{t}}[x^mf(x,t,u,\frac{\partial{u}}{\partial{x}})]+s(x,t,u,\frac{\partial{u}}{\partial{x}})$$

$$p(x,t,u)+q(x,t,u)*f(x,t,u,\frac{\partial{u}}{\partial{x}})=0$$

$$u(x,t_0)=u_0$$

举一个例子：

$$\frac{\partial{u}}{\partial{t}}=\frac{\partial^2{u}}{\partial{x^2}}-u$$

$$u|_{x=0}=1$$

$$u|_{x=1}=0$$

$$u|_{t=0}=(x-1)^2$$

求解过程如下：

~~~matlab
function pde_demo
    x=0:0.05:1;
    t=0:0.05:1;
    m=0;
    sol=pdepe(m,@pdefun,@pdeic,@pdebc,x,t);
    figure('name','pde demo');
    surf(x,t,sol(:,:,1));
    title('pde demo');
    xlabel('x');ylabel('t');zlabel('u');
return

function [c,f,s]=pdefun(x,t,u,du) %方程描述函数
    c=1;
    f=1*du;
    s=-1*u;
return

function [pa,qa,pb,qb]=pdebc(xa,ua,xb,ub,t) %边界描述函数
    pa=ua-1;
    qa=0;
    pb=ub;
    qb=0;
return

function u0=pdeic(x) %初值描述函数
    u0=(x-1)^2;
return
~~~

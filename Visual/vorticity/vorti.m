close all
load wind
load("physical_quantities_data.mat")
% 创建第一个子图（漩涡切片）
% 创建一个新的图形对象
fig = figure;

% 设置图形的背景颜色为纯白色
set(gcf,'color','w'); 
subplot(1, 2, 1);
mycmp=[[ones(20,1),(0.05:0.05:1)',(0.05:0.05:1)'];[(1:-0.05:0.05)',(1:-0.05:0.05)',ones(20,1)]];

%cmap_vorticity = [[ones(20, 1), (0.05:0.05:1)', (0.05:0.05:1)']; [(1:-0.05:0.05)', (1:-0.05:0.05)', ones(20, 1)]];
cav = curl(x, y, z, u, v, w); % 计算旋度
h1 = slice(x, y, z, cav, [-1 1], 0.3, 0.1); % 切片
shading interp
daspect([1 1 1]); % 坐标轴缩放
axis tight
colormap(mycmp); % 应用 "rwb" 颜色映射
caxis([-1, 1]) % 确定颜色范围，使得颜色图中白色对应0涡量
camlight % 设置光照
set([h1(1), h1(2)], 'ambientstrength', .6); % 调整局部亮度

xlabel('X');
ylabel('Y');
zlabel('Z');
title('Vorticity');
% 添加颜色条（colorbar）
colorbar('Location', 'NorthOutside');
% 创建第二个子图（压力场切片）
subplot(1, 2, 2);

cmap_pressure = jet; % 选择不同的颜色映射（这里使用 jet 颜色映射）
h2 = slice(x, y, z, p, [-1 1], 0.3, 0.1); % 切片
shading interp
daspect([1 1 1]); % 坐标轴缩放
axis tight
colormap(cmap_pressure);
caxis([-1, 1]) % 确定颜色范围，使得颜色图中白色对应0压力
camlight % 设置光照
set([h2(1), h2(2)], 'ambientstrength', .6); % 调整局部亮度
% 添加颜色条（colorbar）
colorbar('Location', 'NorthOutside');


xlabel('X');
ylabel('Y');
zlabel('Z');
title('Pressure');

% 设置图标题
sgtitle('Beltrami FLOW (Vorticity and Pressure)');
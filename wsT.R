# 安装和加载必要的包
install.packages("quantmod")
install.packages("rvinecopulib")
install.packages("rugarch")
install.packages("ggplot2")

library(quantmod)
library(rvinecopulib)
library(rugarch)
library(ggplot2)

# 定义要分析的股票
stocks <- c("ZEUS", "X", "RS", "NUE")

# 获取2011年至2021年的历史数据作为训练集
getSymbols(stocks, src = "yahoo", from = "2011-01-01", to = "2021-12-31")
train_prices <- do.call(merge, lapply(stocks, function(x) Cl(get(x))))

# 将数据转换为长格式以便于绘图
train_prices_long <- fortify.zoo(train_prices)
train_prices_long <- reshape2::melt(train_prices_long, id.vars = "Index")

# 使用 ggplot2 绘制四个股票的收盘价格
ggplot(train_prices_long, aes(x = Index, y = value, color = variable)) +
  geom_line() +
  facet_wrap(~ variable, scales = "free_y") +
  labs(title = "Historical Closing Prices of Stocks (2011-2021)",
       x = "Date",
       y = "Closing Price",
       color = "Stock") +
  theme_minimal() +
  theme(legend.position = "none")

# 获取2023年的数据作为测试集
getSymbols(stocks, src = "yahoo", from = "2023-01-01", to = "2023-12-31")
test_prices <- do.call(merge, lapply(stocks, function(x) Cl(get(x))))

# 计算训练集的对数收益率（去除NA值）
train_returns <- na.omit(diff(log(train_prices)))

# 计算测试集的对数收益率（去除NA值）
test_returns <- na.omit(diff(log(test_prices)))
dim(test_returns)

# 合并训练集和测试集的对数收益率
combined_returns <- rbind(train_returns, test_returns)

# 将数据转换为长格式以便于绘图
combined_returns_long <- fortify.zoo(combined_returns)
combined_returns_long <- reshape2::melt(combined_returns_long, id.vars = "Index")

# 使用 ggplot2 绘制四个股票的对数收益率
ggplot(combined_returns_long, aes(x = Index, y = value, color = variable)) +
  geom_line() +
  facet_wrap(~ variable, scales = "free_y") +
  labs(title = "Log Returns of Stocks",
       x = "Date",
       y = "Log Returns",
       color = "Stock") +
  theme_minimal() +
  theme(legend.position = "none")

# 使用 ARMA-GARCH 模型对训练集的边际分布建模
spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                   mean.model = list(armaOrder = c(1, 1), include.mean = TRUE),
                   distribution.model = "std")

standardized_residuals <- list()

for (i in seq_along(stocks)) {
  stock <- stocks[i]
  fit <- ugarchfit(spec, train_returns[, i, drop = FALSE])
  show(fit)
  standardized_residuals[[i]] <- as.numeric(residuals(fit, standardize = TRUE))
}

# 将标准化残差合并为一个矩阵
standardized_residuals_matrix <- do.call(cbind, standardized_residuals)
colnames(standardized_residuals_matrix) <- stocks

# 将标准化残差转换到 [0, 1] 区间
u_data <- pnorm(standardized_residuals_matrix)

# 确保 u_data 是一个矩阵而不是数据框
u_data <- as.matrix(u_data)

# 自动选择最优的 vine copula 结构并进行模型拟合
dvine <- vinecop(data = u_data, family_set = "all")

# 打印 D-vine Copula 模型
print(dvine)
summary(dvine)

# 安装 ggraph 包
install.packages("ggraph")

# 加载 ggraph 包
library(ggraph)

# 绘制 D-vine Copula 模型结构并添加股票名称作为标签
plot(dvine) + 
  geom_node_text(aes(label = stocks), vjust = 1.5, hjust = 1.5)



# 从 D-vine Copula 模型中模拟数据
simulated_data <- rvinecop(dvine, n = 1000)

# 对模拟的数据进行分析，如计算联合分布的值
simulated_returns <- qnorm(simulated_data)

# 可视化模拟的结果
pairs(simulated_returns, main = "Simulated Returns from D-vine Copula Model")



# 单个时间窗口预测代码（仅使用 gamma_window_size）
gamma_window_size <- 100  # 仅用这个窗口

# 创建一个空的向量来存储预测误差
single_window_prediction_errors <- vector("list", length(stocks))
n <- nrow(test_returns)

# 循环处理每一只股票
for (i in seq_along(stocks)) {
  stock <- stocks[i]
  
  pred_errors <- numeric(n - gamma_window_size)
  
  # 对每个时间点进行滚动预测
  for (j in 1:(n - gamma_window_size)) {
    # 定义滚动窗口中的边际模型训练集
    gamma_window <- test_returns[j:(j + gamma_window_size - 1), i, drop = FALSE]
    
    # 使用 ARMA-GARCH 模型对滚动窗口内的训练数据建模
    fit <- tryCatch(
      {
        ugarchfit(spec, gamma_window)
      },
      error = function(e) {
        warning(paste("Error fitting ARMA-GARCH for stock", stock, "at time", j, "-", e$message))
        return(NULL)
      },
      warning = function(w) {
        warning(paste("Warning fitting ARMA-GARCH for stock", stock, "at time", j, "-", w$message))
        return(NULL)
      }
    )
    
    # 如果模型拟合失败，跳过当前迭代
    if (is.null(fit)) {
      next
    }
    
    # 使用 ARMA-GARCH 模型预测下一时刻的收益率
    forecast <- tryCatch(
      {
        ugarchforecast(fit, n.ahead = 1)
      },
      error = function(e) {
        warning(paste("Error forecasting for stock", stock, "at time", j, "-", e$message))
        return(NULL)
      },
      warning = function(w) {
        warning(paste("Warning forecasting for stock", stock, "at time", j, "-", w$message))
        return(NULL)
      }
    )
    
    # 如果预测失败，跳过当前迭代
    if (is.null(forecast)) {
      next
    }
    
    prediction <- forecast@forecast$seriesFor[1]
    
    # 计算实际值和预测值的误差
    actual <- test_returns[j + gamma_window_size, i]
    pred_errors[j] <- actual - prediction
  }
  
  single_window_prediction_errors[[i]] <- pred_errors
}

# 去除 NA 值并创建数据框用于绘图
plot_data_single_window <- data.frame(
  Date = index(test_returns)[(gamma_window_size + 1):n],
  Stock = rep(stocks, each = (n - gamma_window_size)),
  Actual = unlist(lapply(1:length(stocks), function(i) test_returns[(gamma_window_size + 1):n, i])),
  Predicted = unlist(lapply(1:length(stocks), function(i) test_returns[(gamma_window_size + 1):n, i] - single_window_prediction_errors[[i]]))
)

# 去除 NA 值
plot_data_single_window <- na.omit(plot_data_single_window)

# 使用 ggplot2 绘制实际值与单窗口预测值的对比图
ggplot(plot_data_single_window, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  facet_wrap(~ Stock, scales = "free_y") +
  labs(title = "Single Window Prediction: Actual vs Predicted Returns for Each Stock",
       x = "Date",
       y = "Returns") +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  theme_minimal() +
  theme(legend.title = element_blank(), legend.position = "bottom")










# 修改后的预测值计算方式：使用均值加上波动率乘以标准化残差
gamma_window_size <- 100
kappa_window_range <- seq(30, 100, by = 10)

# 初始化误差存储列表
mse_grid <- numeric(length(kappa_window_range))
names(mse_grid) <- kappa_window_range

n <- nrow(test_returns)

# 循环处理每一只股票
for (i in seq_along(stocks)) {
  stock <- stocks[i]
  
  # 检查当前股票的测试数据长度是否足够大
  stock_returns_length <- length(test_returns[, i])
  
  # 滚动交叉验证选择最佳 kappa_window_size
  for (k in seq_along(kappa_window_range)) {
    kappa_window_size <- kappa_window_range[k]
    
    if (stock_returns_length <= gamma_window_size) {
      next
    }
    
    pred_errors <- numeric(n - gamma_window_size)
    
    # 对每个时间点进行滚动预测
    for (j in 1:(n - gamma_window_size)) {
      # 定义滚动窗口中的边际模型训练集
      gamma_window <- test_returns[j:(j + gamma_window_size - 1), i, drop = FALSE]
      
      # 使用 ARMA-GARCH 模型对滚动窗口内的训练数据建模
      fit <- tryCatch(
        {
          ugarchfit(spec, gamma_window)
        },
        error = function(e) {
          warning(paste("Error fitting ARMA-GARCH for stock", stock, "at time", j, "-", e$message))
          return(NULL)
        },
        warning = function(w) {
          warning(paste("Warning fitting ARMA-GARCH for stock", stock, "at time", j, "-", w$message))
          return(NULL)
        }
      )
      
      # 如果模型拟合失败，跳过当前迭代
      if (is.null(fit)) {
        next
      }
      
      # 计算标准化残差
      standardized_residuals <- as.numeric(residuals(fit, standardize = TRUE))
      u_data <- pnorm(standardized_residuals)
      
      # 使用之前拟合好的 dvine copula 模型来模拟标准化残差的联合分布
      if (length(u_data) >= kappa_window_size) {
        kappa_matrix <- matrix(u_data, ncol = 1)
        simulated_data <- rvinecop(dvine, n = 1)
        u_simulated <- simulated_data[, 1]
        
        # 使用 ARMA-GARCH 模型预测下一时刻的收益率
        forecast <- tryCatch(
          {
            ugarchforecast(fit, n.ahead = 1)
          },
          error = function(e) {
            warning(paste("Error forecasting for stock", stock, "at time", j, "-", e$message))
            return(NULL)
          },
          warning = function(w) {
            warning(paste("Warning forecasting for stock", stock, "at time", j, "-", w$message))
            return(NULL)
          }
        )
        
        # 如果预测失败，跳过当前迭代
        if (is.null(forecast)) {
          next
        }
        
        # 提取 ARMA-GARCH 预测的均值和波动率
        mean_forecast <- forecast@forecast$seriesFor[1]
        volatility_forecast <- forecast@forecast$sigmaFor[1]
        
        # 预测值 = 均值 + 波动率 * Copula模拟的u_simulated
        prediction <- mean_forecast + volatility_forecast * u_simulated
        
        # 计算实际值和预测值的误差
        actual <- test_returns[j + gamma_window_size, i]
        pred_errors[j] <- actual - prediction
      }
    }
    
    # 计算当前 kappa_window_size 的均方误差
    mse <- mean(pred_errors^2, na.rm = TRUE)
    mse_grid[k] <- mse
  }
}

# 找到最小 MSE 对应的 kappa_window_size
best_kappa_index <- which.min(mse_grid)
best_kappa <- kappa_window_range[best_kappa_index]

cat("Fixed gamma_window_size:", gamma_window_size, "\n")
cat("Optimal kappa_window_size:", best_kappa, "\n")

# 使用最佳的 kappa_window_size 进行最终的预测
kappa_window_size <- best_kappa

# 使用最佳窗口长度重新进行预测
gamma_window_size <- 100


prediction_errors <- vector("list", length(stocks))

for (i in seq_along(stocks)) {
  stock <- stocks[i]
  
  # 检查当前股票的测试数据长度是否足够大
  stock_returns_length <- length(train_returns[, i])
  print(stock_returns_length)
  if (stock_returns_length <= gamma_window_size) {
    stop(paste("Not enough data for stock", stock, "- increase gamma_window_size or reduce test set size."))
  }
  
  pred_errors <- numeric(n - gamma_window_size)
  
  # 对每个时间点进行滚动预测
  for (j in 1:(n - gamma_window_size)) {
    # 定义滚动窗口中的边际模型训练集
    gamma_window <- test_returns[j:(j + gamma_window_size - 1), i, drop = FALSE]
    
    # 使用 ARMA-GARCH 模型对滚动窗口内的训练数据建模
    fit <- tryCatch(
      {
        ugarchfit(spec, gamma_window)
      },
      error = function(e) {
        warning(paste("Error fitting ARMA-GARCH for stock", stock, "at time", j, "-", e$message))
        return(NULL)
      },
      warning = function(w) {
        warning(paste("Warning fitting ARMA-GARCH for stock", stock, "at time", j, "-", w$message))
        return(NULL)
      }
    )
    
    # 如果模型拟合失败，跳过当前迭代
    if (is.null(fit)) {
      next
    }
    
    # 计算标准化残差
    standardized_residuals <- as.numeric(residuals(fit, standardize = TRUE))
    u_data <- pnorm(standardized_residuals)
    
    # 使用之前拟合好的 dvine copula 模型来模拟标准化残差的联合分布
    if (length(u_data) >= kappa_window_size) {
      kappa_matrix <- matrix(u_data, ncol = 1)
      simulated_data <- rvinecop(dvine, n = 1)
      u_simulated <- simulated_data[, 1]
      
      # 使用 ARMA-GARCH 模型预测下一时刻的收益率
      forecast <- tryCatch(
        {
          ugarchforecast(fit, n.ahead = 1)
        },
        error = function(e) {
          warning(paste("Error forecasting for stock", stock, "at time", j, "-", e$message))
          return(NULL)
        },
        warning = function(w) {
          warning(paste("Warning forecasting for stock", stock, "at time", j, "-", w$message))
          return(NULL)
        }
      )
      
      # 提取 ARMA-GARCH 预测的均值和波动率
      mean_forecast <- forecast@forecast$seriesFor[1]
      volatility_forecast <- forecast@forecast$sigmaFor[1]
      
      # 预测值 = 均值 + 波动率 * Copula模拟的u_simulated
      prediction <- mean_forecast + volatility_forecast * u_simulated
      
      # 计算实际值和预测值的误差
      actual <- test_returns[j + gamma_window_size, i]
      pred_errors[j] <- actual - prediction
    }
  }
  
  prediction_errors[[i]] <- pred_errors
}

# 可视化实际值与预测值
plot_data <- data.frame(Date = index(test_returns)[(gamma_window_size + 1):n],
                        Stock = rep(stocks, each = (n - gamma_window_size)),
                        Actual = unlist(lapply(1:length(stocks), function(i) test_returns[(gamma_window_size + 1):n, i])),
                        Predicted = unlist(lapply(1:length(stocks), function(i) test_returns[(gamma_window_size + 1):n, i] - prediction_errors[[i]])))

# 去除 NA 值
plot_data <- na.omit(plot_data)

# 使用 ggplot2 进行可视化
ggplot(plot_data, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  facet_wrap(~ Stock, scales = "free_y") +
  labs(title = "Double Windows Prediction: Actual vs Predicted Returns for Each Stock",
       x = "Date",
       y = "Returns") +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  theme_minimal() +
  theme(legend.title = element_blank(), legend.position = "bottom")

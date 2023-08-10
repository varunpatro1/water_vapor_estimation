from sklearn.ensemble import RandomForestRegressor


rf = RandomForestRegressor(random_state = 0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

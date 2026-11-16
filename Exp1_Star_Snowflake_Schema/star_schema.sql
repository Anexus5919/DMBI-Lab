-- Star schema for the Rental Car Management System (Experiment 1)
-- One central fact table joined directly to four denormalized dimensions.

CREATE TABLE Dim_Date (
    Date_ID     VARCHAR(10) PRIMARY KEY,
    Date        DATE NOT NULL,
    Day         INT,
    Month       VARCHAR(12),
    Quarter     VARCHAR(2),
    Year        INT
);

CREATE TABLE Dim_Car (
    Car_ID      VARCHAR(10) PRIMARY KEY,
    Car_Model   VARCHAR(40),
    Brand       VARCHAR(30),
    Car_Type    VARCHAR(20),
    Fuel_Type   VARCHAR(15)
);

CREATE TABLE Dim_Customer (
    Customer_ID   VARCHAR(10) PRIMARY KEY,
    Customer_Name VARCHAR(60),
    City          VARCHAR(40),
    State         VARCHAR(40),
    Country       VARCHAR(40)
);

CREATE TABLE Dim_Branch (
    Branch_ID   VARCHAR(10) PRIMARY KEY,
    Branch_Name VARCHAR(60),
    City        VARCHAR(40),
    State       VARCHAR(40)
);

CREATE TABLE Rental_Fact (
    Rental_ID     VARCHAR(10) PRIMARY KEY,
    Date_ID       VARCHAR(10) REFERENCES Dim_Date(Date_ID),
    Car_ID        VARCHAR(10) REFERENCES Dim_Car(Car_ID),
    Customer_ID   VARCHAR(10) REFERENCES Dim_Customer(Customer_ID),
    Branch_ID     VARCHAR(10) REFERENCES Dim_Branch(Branch_ID),
    Rental_Days   INT,
    Rental_Amount DECIMAL(10, 2),
    Late_Fee      DECIMAL(10, 2)
);

-- Example: revenue by car type and quarter
-- SELECT c.Car_Type, d.Year, d.Quarter, SUM(f.Rental_Amount) AS Revenue
-- FROM Rental_Fact f
-- JOIN Dim_Car c  ON f.Car_ID = c.Car_ID
-- JOIN Dim_Date d ON f.Date_ID = d.Date_ID
-- GROUP BY c.Car_Type, d.Year, d.Quarter;

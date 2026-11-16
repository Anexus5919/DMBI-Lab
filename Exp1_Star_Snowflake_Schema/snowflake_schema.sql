-- Snowflake schema for the Rental Car Management System (Experiment 1)
-- Rental_Fact and Dim_Date stay the same as the star schema.
-- Car details are split into model and brand tables, and the customer and
-- branch locations are split into location, city and state tables.

CREATE TABLE Dim_Date (
    Date_ID     VARCHAR(10) PRIMARY KEY,
    Date        DATE NOT NULL,
    Day         INT,
    Month       VARCHAR(12),
    Quarter     VARCHAR(2),
    Year        INT
);

CREATE TABLE Dim_Brand (
    Brand_ID    VARCHAR(10) PRIMARY KEY,
    Brand_Name  VARCHAR(30)
);

CREATE TABLE Dim_Car_Model (
    Model_ID    VARCHAR(10) PRIMARY KEY,
    Model_Name  VARCHAR(40),
    Category    VARCHAR(20),
    Brand_ID    VARCHAR(10) REFERENCES Dim_Brand(Brand_ID)
);

CREATE TABLE Dim_Car (
    Car_ID          VARCHAR(10) PRIMARY KEY,
    Registration_No VARCHAR(15),
    Model_ID        VARCHAR(10) REFERENCES Dim_Car_Model(Model_ID),
    Fuel_Type       VARCHAR(15)
);

CREATE TABLE Dim_State (
    State_ID    VARCHAR(10) PRIMARY KEY,
    State_Name  VARCHAR(40),
    Country     VARCHAR(40)
);

CREATE TABLE Dim_City (
    City_ID     VARCHAR(10) PRIMARY KEY,
    City_Name   VARCHAR(40),
    State_ID    VARCHAR(10) REFERENCES Dim_State(State_ID)
);

CREATE TABLE Dim_Location (
    Location_ID VARCHAR(10) PRIMARY KEY,
    City_ID     VARCHAR(10) REFERENCES Dim_City(City_ID)
);

CREATE TABLE Dim_Customer (
    Customer_ID   VARCHAR(10) PRIMARY KEY,
    Customer_Name VARCHAR(60),
    Location_ID   VARCHAR(10) REFERENCES Dim_Location(Location_ID)
);

CREATE TABLE Dim_Branch (
    Branch_ID   VARCHAR(10) PRIMARY KEY,
    Branch_Name VARCHAR(60),
    Location_ID VARCHAR(10) REFERENCES Dim_Location(Location_ID)
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

-- Revenue by brand now needs two extra joins compared with the star schema:
-- SELECT b.Brand_Name, SUM(f.Rental_Amount) AS Revenue
-- FROM Rental_Fact f
-- JOIN Dim_Car c        ON f.Car_ID = c.Car_ID
-- JOIN Dim_Car_Model m  ON c.Model_ID = m.Model_ID
-- JOIN Dim_Brand b      ON m.Brand_ID = b.Brand_ID
-- GROUP BY b.Brand_Name;

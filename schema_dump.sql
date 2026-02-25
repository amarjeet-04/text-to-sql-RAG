CREATE TABLE AgentMaster_V1 (
  Id INT NOT NULL,
  AgentCode NVARCHAR NULL,
  AgentName NVARCHAR NOT NULL,
  SubUserCode NVARCHAR NULL,
  TotalAgentFare INT NOT NULL,
  AgentId INT NOT NULL,
  UserCreate NVARCHAR NULL,
  AgentCountry NVARCHAR NULL,
  AgentRegion VARCHAR NOT NULL,
  AgentState VARCHAR NOT NULL,
  AgentCity NVARCHAR NULL,
  AgentAreaBranch NVARCHAR NULL,
  AgentPinCode NVARCHAR NOT NULL,
  AgentCategory VARCHAR NULL,
  CreditLimit MONEY NOT NULL,
  CreditLimitUpdatedBy NVARCHAR NOT NULL,
  CreditlimitUpdateOn DATETIME NOT NULL,
  parentID INT NULL,
  SubUserId INT NULL,
  SubUserName NVARCHAR NULL,
  SalesPersonId INT NULL,
  SalesPersonName NVARCHAR NULL,
  AgentType NVARCHAR NULL,
  Type NVARCHAR NULL,
  Status NVARCHAR NULL,
  PhoneNo VARCHAR NULL,
  EmailId VARCHAR NULL
)

CREATE TABLE AgentTypeMapping (
  AgentID INT NULL,
  AgentName NVARCHAR NULL,
  AgentCode NVARCHAR NULL,
  AgentType VARCHAR NULL,
  StatusFlag INT NULL
)

CREATE TABLE Agent_Country_Master (
  AgentCountry NVARCHAR NULL,
  Region NVARCHAR NULL
)

CREATE TABLE BookingData (
  MasterId INT NULL,
  DetailId INT NOT NULL,
  PNRNo NVARCHAR NULL,
  ProductCountryid INT NULL,
  ProductCityId INT NULL,
  AgentId INT NULL,
  SubUserId INT NULL,
  CreditcardCharges DECIMAL NULL,
  ClientNatinality VARCHAR NULL,
  PaymentType NVARCHAR NULL,
  IsPackage INT NULL,
  BookingStatus NVARCHAR NULL,
  AgentReferenceNo NVARCHAR NULL,
  CurrencyId INT NULL,
  AgentBuyingPrice DECIMAL NULL,
  AgentSellingPrice DECIMAL NULL,
  IsSameCurrency INT NULL,
  SupplierId INT NULL,
  RateOfexchange DECIMAL NULL,
  SellingRateOfexchange DECIMAL NULL,
  CompanyBuyingPrice DECIMAL NULL,
  SubAgentSellingPrice DECIMAL NULL,
  SupplierRateOfexchange DECIMAL NULL,
  SupplierCurrencyId INT NULL,
  CheckInDate DATETIME NULL,
  CheckOutDate DATETIME NULL,
  RoomTypeName NVARCHAR NULL,
  Provider NVARCHAR NULL,
  OfferCode NVARCHAR NULL,
  OfferDescription VARCHAR NULL,
  CancellationDeadLine DATETIME NULL,
  LoyaltyPoints INT NULL,
  OTHPromoCode NVARCHAR NULL,
  IsXMLSupplierBooking INT NOT NULL,
  ProductId INT NULL,
  ProductName NVARCHAR NULL,
  ServiceName VARCHAR NOT NULL,
  CreatedDate DATETIME NULL,
  BranchId INT NULL,
  NoofAdult INT NULL,
  NoofChild INT NULL,
  PackageId INT NULL,
  PackageName NVARCHAR NULL,
  EntryDate DATETIME NULL
)

CREATE TABLE HotelChain (
  HotelId INT NULL,
  HotelName NVARCHAR NULL,
  Country NVARCHAR NULL,
  City NVARCHAR NULL,
  Star NVARCHAR NULL,
  Chain NVARCHAR NULL
)

CREATE TABLE Master_City (
  CountryID NVARCHAR NULL,
  Country NVARCHAR NULL,
  CityID NVARCHAR NULL,
  City NVARCHAR NULL
)

CREATE TABLE Master_Country (
  CountryID NVARCHAR NULL,
  Country NVARCHAR NULL
)

CREATE TABLE suppliermaster_Report (
  EmployeeId INT NULL,
  EmployeeCode NVARCHAR NULL,
  FirstName NVARCHAR NULL,
  LocalAddress NVARCHAR NULL,
  MobileNumber NVARCHAR NULL,
  EmailId NVARCHAR NULL,
  SupplierName NVARCHAR NULL,
  Status BIT NULL,
  AuthStatus BIT NULL,
  SupplierType NVARCHAR NULL,
  Country NVARCHAR NULL,
  City NVARCHAR NULL
)
SELECT * FROM dbo.IMAGE_DATA

-- Create
INSERT INTO dbo.IMAGE_DATA(URL,LABEL)
VALUES ('thisisatest.com', 'not comet')
SELECT * FROM dbo.IMAGE_DATA

-- Read
DECLARE @url VARCHAR(100)
set @url = (SELECT TOP 1 URL FROM dbo.IMAGE_DATA WHERE URL = 'thisisatest.com')
PRINT @url

SELECT * FROM dbo.IMAGE_DATA

-- Update
UPDATE dbo.IMAGE_DATA
SET LABEL = 'comet'
WHERE URL = 'thisisatest.com'

SELECT * FROM dbo.IMAGE_DATA

-- Destroy
DELETE FROM dbo.IMAGE_DATA
WHERE URL = 'thisisatest.com'

SELECT * FROM dbo.IMAGE_DATA
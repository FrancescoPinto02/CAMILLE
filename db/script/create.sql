DROP SCHEMA IF EXISTS camille_db;
CREATE DATABASE IF NOT EXISTS camille_db;
USE camille_db;

CREATE TABLE pipeline_stage (
    id INT AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    
    PRIMARY KEY (id)
);

CREATE TABLE effect (
    id INT AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    
    PRIMARY KEY (id)
);

CREATE TABLE codesmell (
    id INT AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    problems TEXT NOT NULL,
    solution TEXT NOT NULL,
    type ENUM('Generic', 'API-Specific') NOT NULL,
    
    PRIMARY KEY (id)
);

CREATE TABLE codesmell_stage (
    codesmell_id INT NOT NULL,
    stage_id INT NOT NULL,
    
    FOREIGN KEY (codesmell_id) REFERENCES codesmell(id) 
		ON DELETE CASCADE 
        ON UPDATE CASCADE,
    FOREIGN KEY (stage_id) REFERENCES pipeline_stage(id) 
		ON DELETE CASCADE 
        ON UPDATE CASCADE,
    PRIMARY KEY (codesmell_id, stage_id)
);

CREATE TABLE codesmell_effect (
    codesmell_id INT NOT NULL,
    effect_id INT NOT NULL,
    
    FOREIGN KEY (codesmell_id) REFERENCES codesmell(id) 
		ON DELETE CASCADE 
        ON UPDATE CASCADE,
    FOREIGN KEY (effect_id) REFERENCES effect(id) 
		ON DELETE CASCADE 
        ON UPDATE CASCADE,
    PRIMARY KEY (codesmell_id, effect_id)
);

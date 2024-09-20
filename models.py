from sqlalchemy import Column, Integer, String, Text, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class StudentAnswer(Base):
    __tablename__ = "se_course_student_file"
    nid = Column(Integer, primary_key=True, index=True, autoincrement=True)
    nid_section_file = Column(Integer)
    npm = Column(String(12))
    cuserid = Column(String(12))
    cfile = Column(Text)
    pertemuan = Column(String(255))
    cextension = Column(String(5))
    cpath = Column(String(255))
    dupload = Column(DateTime)
    npoint = Column(Float)
    cacademic_year = Column(String(11))

class LecturerAnswer(Base):
    __tablename__ = "se_upload_dosen"
    nid = Column(Integer, primary_key=True, index=True, autoincrement=True)
    nid_section_file = Column(Text)
    cuserid = Column(String(12))
    cfile = Column(String(255))
    answer_text = Column(Text)
    pertemuan = Column(Integer)
    cextension = Column(String(5))
    cpath = Column(String(255))
    cacademic_year = Column(String(11))
    dupload = Column(DateTime)
    npoint = Column(Float)

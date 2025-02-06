import React, { useState } from "react";
import '../index.css';
import SubmitButton from "./SubmitButton";

const KeywordInput = () => {
  const [keyword, setKeyword] = useState(""); // 키워드 상태 관리

  const handleInputChange = (e) => {
    setKeyword(e.target.value); // 입력값을 상태에 저장
  };

  const handleSubmit = () => {
    console.log("입력된 키워드:", keyword); // 키워드를 출력 (백엔드로 전달 예정)
    // 여기서 API 호출 등 추가 작업을 수행할 수 있음
  };

  return (
    <div className="keyword-input-container">
      <input
        type="text"
        placeholder="키워드를 입력하세요"
        value={keyword}
        onChange={handleInputChange}
        style={{
          padding: "10px",
          width: "300px",
          marginRight: "10px",
          borderRadius: "5px",
          border: "1px solid #ccc",
        }}
      />
      <SubmitButton />
    </div>
  );
};

export default KeywordInput;